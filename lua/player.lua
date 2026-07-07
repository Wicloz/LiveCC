-- LiveCC player  —  watches YouTube videos / live streams on a CC monitor (or the
--                   computer's own terminal when none is attached, e.g. a tablet)
--                   plays audio on ALL attached speakers simultaneously, unless
--                   a livecc.speakers channel map says otherwise (see below)
-- Run `livecc --help` for usage.
--
-- One WebSocket carries everything, speaking CCMF (the ComputerCraft Media
-- Format, docs/cc-media-format.md).  Every message is binary: a message whose
-- first byte is the container marker (67, "C") is a media chunk — a video GOP
-- (palette + raw/delta/repeat frame units) or ~1 s of audio — and anything
-- else is a control frame [opcode][len u16][body]: STATUS, ERROR.
-- The client opens with ROOM -> ACK, CAPS -> ACK, START, then just consumes.
--
-- Synchronization (spec §5.6): ONE media clock, anchored by the server's
-- STATUS playing message (it carries the origin PTS that is due "now") and
-- re-anchored whenever the server re-sends it (rebuffer, live-edge skip).
-- EVERYTHING presents against that clock by PTS: video frames render when
-- due, and audio — every channel role — is fed to the speakers when due, not
-- when it arrives.  The server delivers chunks a second or two early on
-- purpose; that lead is jitter slack to keep queued here, never a cue to play
-- sooner.  This is what keeps audio on video and every speaker on every other
-- speaker: they all follow the same clock, so none of them can drift.

local DEFAULT_FPS = 24

-- Baked in by the server when this script is installed (GET /player).
local SERVER = "{{SERVER}}"

local args = { ... }

local function die(msg)
    printError(msg)
    error()
end

local function print_usage()
    print("livecc - play videos and live streams on a ComputerCraft monitor or terminal")
    print("")
    print("Usage:")
    print("  livecc <url> [options]")
    print("")
    print("Arguments:")
    print("  url           Video or live-stream URL (anything yt-dlp supports)")
    print("")
    print("Options:")
    print("  --fps N       Target frame rate (default: " .. DEFAULT_FPS .. ")")
    print("  --start TS    Start a VOD at timestamp TS")
    print("  --end TS      Stop a VOD at timestamp TS")
    print("  --loop        Loop the selected section forever")
    print("  --sync        Sync playback with other --sync clients on this URL")
    print("  --no-audio    Don't stream audio (also implied when no speakers)")
    print("  --no-video    Don't stream video (audio only)")
    print("  --verbose     Print each chunk's header as it arrives")
    print("  -h, --help    Show this help and exit")
    print("")
    print("Timestamps (TS): 90, 90s, 1m30s, 3h2m  (default unit: seconds)")
    print("")
    print("Speakers: every speaker plays the same mono audio by default. To assign")
    print("speakers to channels instead, put a livecc.speakers file next to this")
    print("script: a Lua table { [peripheralName] = \"roleName\" }, e.g.")
    print("  { left = \"front_left\", right = \"front_right\" }")
    print("Unlisted speakers stay silent. Roles: mono, front_left, front_right,")
    print("center, lfe, surround_left, surround_right, rear_left, rear_right.")
    print("See lua/livecc.speakers.example in the repo for a starter file.")
    print("")
    print("Examples:")
    print("  livecc https://youtu.be/dQw4w9WgXcQ")
    print("  livecc https://youtu.be/XYZ --fps 30")
    print("  livecc https://youtu.be/XYZ --start 1m30s --loop")
end

-- Parse: one positional (url) plus optional flags.
local positional = {}
local START_TS, END_TS, LOOP, HELP, SYNC = nil, nil, false, false, false
local NO_AUDIO, NO_VIDEO, VERBOSE = false, false, false
local WANTED_FPS = DEFAULT_FPS
do
    local i = 1
    while i <= #args do
        local a = args[i]
        if a == "-h" or a == "--help" then
            HELP = true
        elseif a == "--fps" then
            i = i + 1; WANTED_FPS = tonumber(args[i]) or DEFAULT_FPS
        elseif a == "--start" then
            i = i + 1; START_TS = args[i]
        elseif a == "--end" then
            i = i + 1; END_TS = args[i]
        elseif a == "--loop" then
            LOOP = true
        elseif a == "--sync" then
            SYNC = true
        elseif a == "--no-audio" then
            NO_AUDIO = true
        elseif a == "--no-video" then
            NO_VIDEO = true
        elseif a == "--verbose" then
            VERBOSE = true
        else
            positional[#positional + 1] = a
        end
        i = i + 1
    end
end

if HELP then
    print_usage()
    return
end

if #positional < 1 then
    printError("livecc: missing <url>")
    print("Run 'livecc --help' for usage.")
    return
end

if SERVER:find("^{{") then   -- placeholder never replaced => not installed via server
    die("livecc isn't linked to a server. Reinstall with: wget run <server>/install")
end

local VIDEO_URL = positional[1]

-- Timestamps ("90", "90s", "1m30s", "3h2m") -> milliseconds (the ROOM unit),
-- or nil for empty/unrecognised input.
local function parse_timestamp_ms(value)
    if not value or value == "" then return nil end
    value = value:lower()
    if value:match("^%d+$") then return tonumber(value) * 1000 end
    if not value:match("^%d") then return nil end
    local h = tonumber(value:match("(%d+)h") or 0)
    local m = tonumber(value:match("(%d+)m") or 0)
    local s = tonumber(value:match("(%d+)s") or 0)
    local total = (h * 3600 + m * 60 + s) * 1000
    if total == 0 then return nil end
    return total
end

local START_MS = parse_timestamp_ms(START_TS)
local END_MS = parse_timestamp_ms(END_TS)

-- ── Display + speakers ──────────────────────────────────────────────────────────
-- Render to an attached monitor if there is one; otherwise fall back to the
-- computer's own terminal (e.g. a pocket computer / tablet with no monitor).  Both
-- expose the same drawing API (blit/setCursorPos/getSize/...); only setTextScale
-- is monitor-only.

local mon = peripheral.find("monitor")
local USING_TERM = mon == nil
if USING_TERM then mon = term end

-- When the render surface IS our own terminal there's no separate console, so
-- silence the status print()s — otherwise they'd scribble over the video.
-- mon_print() still shows the important states (buffering/error/...) on-screen.
local function console(msg)
    if not USING_TERM then print(msg) end
end

if mon.setTextScale then mon.setTextScale(0.5) end
mon.setCursorBlink(false)            -- no blinking cursor in the corner
local TERM_W, TERM_H = mon.getSize()

-- Draw everything through a buffered window so a GOP's palette unit AND its
-- frame's cell blits reach the monitor in ONE redraw (setVisible(false) -> draw ->
-- setVisible(true) flushes the window's palette and contents together).  Applied
-- straight to the monitor, a palette change would instantly recolour the PREVIOUS
-- frame still on screen — a near-monochrome/inverted ghost flash if a monitor
-- refresh lands between the palette call and the blits.  Buffering makes it atomic.
local screen = window.create(mon, 1, 1, TERM_W, TERM_H, true)

-- Optional per-speaker channel map: a file (SPEAKER_MAP_FILE) containing a Lua
-- table literal { [peripheralName] = "roleName", ... }, e.g.
--   { left = "front_left", right = "front_right", sub = "lfe" }
-- (see livecc.speakers.example in this directory for a starter file).
-- Role names match the CCMF spec's channel roles (docs/cc-media-format.md §4.6).
-- Absent, empty, or malformed -> nil, meaning the original/default strategy:
-- every speaker plays the mono mix.  With a map active, a speaker with no
-- entry in it is left out of every role entirely -- it stays silent, rather
-- than guessing it should get the mono mix too.
local SPEAKER_MAP_FILE = "livecc.speakers"
local ROLE_IDS = {
    mono = 0, front_left = 1, front_right = 2, center = 3, lfe = 4,
    surround_left = 5, surround_right = 6, rear_left = 7, rear_right = 8,
}
local ROLE_NAMES = {}
for name, id in pairs(ROLE_IDS) do ROLE_NAMES[id] = name end

local function load_speaker_map()
    if not fs.exists(SPEAKER_MAP_FILE) then return nil end
    local f = fs.open(SPEAKER_MAP_FILE, "r")
    local content = f.readAll()
    f.close()
    local ok, map = pcall(textutils.unserialize, content)
    if not ok or type(map) ~= "table" then
        console("LiveCC: " .. SPEAKER_MAP_FILE .. " is malformed - ignoring (mono on all speakers)")
        return nil
    end
    local resolved = {}
    for name, role in pairs(map) do
        local id = ROLE_IDS[role]
        if id then
            resolved[name] = id
        else
            console("LiveCC: " .. SPEAKER_MAP_FILE .. ": unknown role '" .. tostring(role)
                .. "' for '" .. tostring(name) .. "' - ignoring that entry")
        end
    end
    return next(resolved) and resolved or nil
end

local speaker_map = load_speaker_map()

-- One physical speaker can be reachable under two names (e.g. adjacent AND via
-- a wired modem).  Driving it twice plays every audio segment twice.  Two wraps
-- of the same speaker share one audio buffer, so we detect duplicates: fill one
-- buffer with silence and check whether the other then reports full.
--
-- Skipped entirely when a speaker map is active: the map names peripherals
-- explicitly, so we trust it outright rather than second-guess it -- e.g. an
-- operator who deliberately lists the same physical speaker under two roles
-- gets exactly that (driving it twice is their call now, not something to
-- silently collapse).
local function dedupe_speakers(found)
    if #found < 2 then return found end
    local silence = {}
    for i = 1, 8192 do silence[i] = 0 end
    local unique = {}
    for _, s in ipairs(found) do
        local dup = false
        for _, u in ipairs(unique) do
            u.dev.stop(); s.dev.stop()
            local guard = 0
            while u.dev.playAudio(silence) and guard < 64 do guard = guard + 1 end
            if not s.dev.playAudio(silence) then dup = true end   -- shares u's buffer
            u.dev.stop(); s.dev.stop()
            if dup then break end
        end
        if not dup then unique[#unique + 1] = s end
    end
    for _, u in ipairs(unique) do u.dev.stop() end
    return unique
end

local found_speakers = {}
for _, name in ipairs(peripheral.getNames()) do
    if peripheral.getType(name) == "speaker" then
        found_speakers[#found_speakers + 1] = { name = name, dev = peripheral.wrap(name) }
    end
end
local deduped_speakers = speaker_map and found_speakers or dedupe_speakers(found_speakers)

-- `speakers`: flat device list of every speaker found (used for the status
-- message/count below).  `speakers_by_role`: role id -> device list — the ONE
-- shape the audio engine drives.  No map: every speaker sits under role 0
-- (mono), the original default.  With a map: exactly the mapped assignments;
-- an unlisted speaker gets no role at all, so it stays silent.
local speakers = {}
local speakers_by_role = {}
for _, s in ipairs(deduped_speakers) do
    speakers[#speakers + 1] = s.dev
    if speaker_map then
        local role = speaker_map[s.name]
        if role then
            speakers_by_role[role] = speakers_by_role[role] or {}
            table.insert(speakers_by_role[role], s.dev)
        end
    end
end
if not speaker_map and #speakers > 0 then
    speakers_by_role[0] = speakers
end

-- A stream is on unless disabled by flag; audio additionally needs a speaker, so
-- the no-speaker case folds into --no-audio (which is what fixes the old hang:
-- the audio coroutine no longer joins parallel.waitForAny when there's no audio).
local WANT_VIDEO = not NO_VIDEO
local WANT_AUDIO = not NO_AUDIO and #speakers > 0

if NO_AUDIO then
    console("LiveCC: audio disabled (--no-audio)")
elseif #speakers == 0 then
    console("LiveCC: no speakers found - audio disabled")
else
    local msg = "LiveCC: " .. #speakers .. " speaker(s)"
    if #found_speakers > #speakers then
        msg = msg .. " (" .. (#found_speakers - #speakers) .. " duplicate ignored)"
    end
    if speaker_map then
        msg = msg .. ", channel map from " .. SPEAKER_MAP_FILE
    end
    console(msg)
    if speaker_map then
        local mapped_total = 0
        for _, list in pairs(speakers_by_role) do mapped_total = mapped_total + #list end
        if mapped_total == 0 then
            console("LiveCC: warning - no attached speaker matches any entry in "
                .. SPEAKER_MAP_FILE .. "; every speaker will stay silent")
        elseif mapped_total < #speakers then
            console("LiveCC: " .. (#speakers - mapped_total)
                .. " unlisted speaker(s) will stay silent")
        end
    end
end

if not WANT_VIDEO and not WANT_AUDIO then
    die("Nothing to play: video is off (--no-video) and there's no audio.")
end

-- ── CCMF byte helpers ────────────────────────────────────────────────────────────
-- All integers are unsigned little-endian (spec §3).  Lua 5.1 has no bitwise
-- operators, so everything is arithmetic; PTS values (u48) stay well inside the
-- double's 2^53 exact-integer range.

local floor = math.floor

local function u16(s, i) return s:byte(i) + s:byte(i + 1) * 256 end

local function u24(s, i)
    return s:byte(i) + s:byte(i + 1) * 256 + s:byte(i + 2) * 65536
end

local function u48(s, i)
    local b1, b2, b3, b4, b5, b6 = s:byte(i, i + 5)
    return b1 + b2 * 2 ^ 8 + b3 * 2 ^ 16 + b4 * 2 ^ 24 + b5 * 2 ^ 32 + b6 * 2 ^ 40
end

local function pack_u16(v)
    return string.char(v % 256, floor(v / 256) % 256)
end

local function pack_u32(v)
    return string.char(v % 256, floor(v / 256) % 256,
                       floor(v / 65536) % 256, floor(v / 16777216) % 256)
end

-- Container / stream constants (spec §4, §5).
local MARKER = 67                        -- "C": leads every media chunk
local TYPE_VIDEO, TYPE_AUDIO = 0, 1
local OP_ROOM, OP_CAPS, OP_START, OP_QUIT = 1, 2, 3, 4
local OP_ACK, OP_ERROR, OP_STATUS = 5, 6, 7

local function control_frame(opcode, body)
    body = body or ""
    return string.char(opcode) .. pack_u16(#body) .. body
end

local function room_body()
    local flags = (LOOP and 1 or 0) + (SYNC and 2 or 0)
        + (START_MS and 4 or 0) + (END_MS and 8 or 0)
    return string.char(flags)
        .. (START_MS and pack_u32(START_MS) or "")
        .. (END_MS and pack_u32(END_MS) or "")
        .. VIDEO_URL
end

-- channels: bit0 (mono) is always advertised — it's the mandatory fallback
-- (spec §5.4) — plus one bit per positional role a speaker map assigns.  This
-- mask declares the SPEAKER LAYOUT, not a wish list: the server serves every
-- advertised role whatever the source's own layout is, deriving roles the
-- source lacks from the nearest mix (ultimately mono, spec §5.4) — so a
-- mapped speaker always plays, it just may carry a downmix/duplicate.
local function channels_mask()
    local mask = 1
    for role in pairs(speakers_by_role) do
        if role > 0 then mask = mask + 2 ^ role end
    end
    return mask
end

local function caps_body()
    local flags = (WANT_VIDEO and 1 or 0) + (WANT_AUDIO and 2 or 0)
    local fps = math.max(1, math.min(255, floor(WANTED_FPS)))
    return string.char(flags)
        .. string.char(0)                -- video: no optional encodings yet
        .. string.char(3)                -- audio: pcm8 + dfpwm both decodable
        .. pack_u16(channels_mask())
        .. string.char(1)                -- compression: none only
        .. pack_u16(TERM_W) .. pack_u16(TERM_H) .. string.char(fps)
end

local ws = nil

-- ── The media clock ──────────────────────────────────────────────────────────────
-- One clock for everything, in 48 kHz samples: PTS `anchor_pts` is due at real
-- time `anchor_ms` (os.epoch "utc").  The server anchors it explicitly with
-- STATUS playing (+ origin PTS, spec §5.6) at start and at every discontinuity
-- (rebuffer resume, live-edge skip); a draft-00 server without origins falls
-- back to anchoring on the first media chunk.  CLIENT_DELAY_MS is fixed
-- startup slack so the first frames/samples are in hand before they're due —
-- the same constant on every client, so --sync rooms stay aligned.

local CLIENT_DELAY_MS = 300
local anchor_pts, anchor_ms = nil, nil

local function clock_now()
    if not anchor_pts then return nil end
    return anchor_pts + (os.epoch("utc") - anchor_ms) * 48   -- 48 samples per ms
end

local function due_ms(pts)
    return anchor_ms + (pts - anchor_pts) / 48
end

-- ── Audio: per-role queues + clock-paced speaker feed ───────────────────────────
-- Chunks land in a per-role queue as RAW payload strings tagged with PTS, and
-- a single engine feeds every role's speakers against the clock:
--
--   * a chunk is fed when it is DUE (its PTS reaches the clock) — the speaker
--     then plays it immediately, so audible time ≈ clock time;
--   * once a role's pipeline is rolling (the speaker holds future samples),
--     contiguous chunks are fed up to SPK_AHEAD early so the device stays
--     buffered across event-loop hiccups; the speaker's own refusal
--     (playAudio -> false, retried on speaker_audio_empty) self-clocks the
--     rest — samples play back-to-back at 48 kHz, in step with the clock;
--   * a chunk whose time has entirely passed is DROPPED, and one that
--     straddles "now" is trimmed, so a role that fell behind (or just joined)
--     re-enters exactly on the clock instead of late;
--   * a HOLE in a role's stream stops the feed until the device drains, then
--     the due-gate restarts it on time — the gap is heard as silence, and the
--     role comes back aligned, never shifted.
--
-- Every role follows the same discipline against the same clock, which is
-- what keeps stereo/5.1/7.1 speakers sample-aligned with each other: nothing
-- about one role's delivery, drops, or stalls can shift another role, and
-- none of them can walk away from video.
--
-- Decoding happens at feed time (pcm8 unpack, or a fresh DFPWM decoder per
-- chunk — its state MUST reset per chunk, spec §4.6): queued chunks stay as
-- compact strings instead of huge Lua number tables.

-- Safety cap against a stalled/refusing role, not a latency knob: comfortably
-- above the server's worst-case VOD catch-up burst (VOD_MAX_BUFFER 15 s /
-- AUDIO_CHUNK_SECONDS 1.0 s = 15 chunks backpressured, then flushed at once
-- on resume), so normal bursty delivery never legitimately overflows it.
local MAX_AUDIO_CHUNKS = 16
local SPK_AHEAD = 38400                -- feed ≤0.8 s ahead once a role is rolling
local PTS_SLOP = 240                   -- 5 ms: chunks this close count as contiguous

local audio_queues = {}                -- role -> { {pts, n, data, codec}, ... }
local role_state = {}                  -- role -> { fed_until, pending }
for role in pairs(speakers_by_role) do
    audio_queues[role] = {}
    role_state[role] = {}
end

local dfpwm = require("cc.audio.dfpwm")

local function decode_pcm8(data)
    local out, k, n = {}, 0, #data
    for i = 1, n, 4096 do
        local bytes = { data:byte(i, math.min(i + 4095, n)) }
        for m = 1, #bytes do
            k = k + 1
            out[k] = bytes[m] - 128
        end
    end
    return out
end

local function decode_samples(head)
    if head.codec == 0 then
        return decode_pcm8(head.data)
    end
    return dfpwm.make_decoder()(head.data)   -- fresh state per chunk (spec §4.6)
end

local function trim_front(samples, skip)
    if skip <= 0 then return samples end
    local out = {}
    for i = skip + 1, #samples do out[i - skip] = samples[i] end
    return out
end

local function handle_audio(pts, payload)
    local hdr = payload:byte(1)
    local codec, channel = floor(hdr / 16), hdr % 16
    local queue = audio_queues[channel]  -- nil if no speaker plays this role
    if not queue or codec > 1 then return end
    local data = payload:sub(2)
    queue[#queue + 1] = {
        pts = pts,
        n = codec == 1 and #data * 8 or #data,   -- dfpwm: 8 samples/byte
        data = data,
        codec = codec,
    }
    while #queue > MAX_AUDIO_CHUNKS do
        table.remove(queue, 1)           -- overflow: drop oldest (PTS re-syncs)
    end
    os.queueEvent("livecc_audio")        -- wake the audio engine
end

-- One pass over every role: retry refused feeds, drop stale, feed what the
-- gates allow.  Returns how many ms until the next scheduled feed (for a
-- wake-up timer), or nil to sleep until an event.
local function feed_roles()
    local now = clock_now()
    if not now then return nil end
    local wake = nil
    local function sooner(ms)
        if not wake or ms < wake then wake = ms end
    end

    for role, devs in pairs(speakers_by_role) do
        local st = role_state[role]
        -- Per-device feed order must hold, so a role with refused devices
        -- retries those first and feeds nothing new until they clear.
        if st.pending then
            local left = {}
            for _, d in ipairs(st.pending.devs) do
                if not d.playAudio(st.pending.samples) then left[#left + 1] = d end
            end
            if #left > 0 then
                st.pending.devs = left
                sooner(100)              -- safety net besides speaker_audio_empty
            else
                st.pending = nil
            end
        end
        local q = audio_queues[role]
        while not st.pending and q[1] do
            now = clock_now()
            local head = q[1]
            if head.pts + head.n <= now then
                table.remove(q, 1)       -- stale: its whole span already passed
            else
                local live = st.fed_until ~= nil and st.fed_until > now
                if live and head.pts > st.fed_until + PTS_SLOP then
                    -- Hole in this role's stream: let the device drain, then
                    -- the due-gate below restarts exactly on the clock.
                    sooner((st.fed_until - now) / 48)
                    break
                elseif live and head.pts > now + SPK_AHEAD then
                    sooner((head.pts - SPK_AHEAD - now) / 48)     -- budget gate
                    break
                elseif not live and head.pts > now then
                    sooner((head.pts - now) / 48)                 -- due gate
                    break
                end
                table.remove(q, 1)
                local samples = decode_samples(head)
                if not live and head.pts < now then
                    samples = trim_front(samples, floor(now - head.pts))
                end
                if #samples > 0 then
                    local refused = {}
                    for _, d in ipairs(devs) do
                        if not d.playAudio(samples) then refused[#refused + 1] = d end
                    end
                    if #refused > 0 then
                        st.pending = { samples = samples, devs = refused }
                        sooner(100)
                    end
                end
                st.fed_until = head.pts + head.n
            end
        end
    end
    return wake
end

local function audio_pending()
    for role, q in pairs(audio_queues) do
        if #q > 0 then return true end
        if role_state[role].pending then return true end
    end
    return false
end

-- Drop everything queued/buffered so playback can restart cleanly at a new
-- clock position (speaker.stop() flushes the device's buffered samples).
local function flush_speakers()
    for role, devs in pairs(speakers_by_role) do
        for _, d in ipairs(devs) do pcall(d.stop) end
        role_state[role].pending = nil
        role_state[role].fed_until = nil
    end
end

-- ── Video: unit queue + PTS pacing ──────────────────────────────────────────────
-- A video chunk is a self-contained GOP: [w u16][h u16] then a
-- unit stream — a 48-byte palette unit, a raw keyframe, and delta/repeat units,
-- each frame carrying its hold duration in 48 kHz samples.  Arriving units land
-- in `vqueue` with an absolute PTS, and the player coroutine renders whatever
-- is due against the media clock.

local MAX_VIDEO_UNITS = 400            -- OOM guard only; the server's lead is ~2 s

local vqueue = {}

local HEXB = {}                          -- palette index 0-15 -> blit hex char byte
for i = 0, 15 do HEXB[i] = ("0123456789abcdef"):byte(i + 1) end
local last_palette = nil                 -- last applied palette block (skip no-ops)

local function apply_palette(pal)
    if pal == last_palette then return end
    if screen.setPaletteColour then
        for i = 0, 15 do
            local o = i * 3 + 1
            screen.setPaletteColour(2 ^ i, pal:byte(o) / 255,
                pal:byte(o + 1) / 255, pal:byte(o + 2) / 255)
        end
    end
    last_palette = pal
end

-- Render a raw keyframe: chars plane (8 five-bit codes per 5 bytes, MSB-first),
-- then the fg and bg nibble planes (2 cells/byte, high nibble first).
local function render_raw(data, W, H)
    local n = W * H
    local unpack = table.unpack or unpack

    local text = {}
    local p, ci = 1, 0
    while ci < n do
        local b1, b2, b3, b4, b5 = data:byte(p, p + 4)
        p = p + 5
        text[ci + 1] = 128 + floor(b1 / 8)
        text[ci + 2] = 128 + (b1 % 8) * 4 + floor(b2 / 64)
        text[ci + 3] = 128 + floor(b2 / 2) % 32
        text[ci + 4] = 128 + (b2 % 2) * 16 + floor(b3 / 16)
        text[ci + 5] = 128 + (b3 % 16) * 2 + floor(b4 / 128)
        text[ci + 6] = 128 + floor(b4 / 4) % 32
        text[ci + 7] = 128 + (b4 % 4) * 8 + floor(b5 / 32)
        text[ci + 8] = 128 + b5 % 32
        ci = ci + 8
    end
    p = math.ceil(n / 8) * 5 + 1

    local function nibble_plane()
        local out, i = {}, 0
        while i < n do
            local b = data:byte(p); p = p + 1
            i = i + 1; out[i] = HEXB[floor(b / 16)]
            if i < n then
                i = i + 1; out[i] = HEXB[b % 16]
            end
        end
        return out
    end
    local fg = nibble_plane()
    local bg = nibble_plane()

    for y = 1, H do
        local base = (y - 1) * W
        screen.setCursorPos(1, y)
        screen.blit(string.char(unpack(text, base + 1, base + W)),
                    string.char(unpack(fg, base + 1, base + W)),
                    string.char(unpack(bg, base + 1, base + W)))
    end
end

-- Render a delta frame: spans of changed cells, blitted in place.  Untouched
-- cells persist in the (buffered) window, so no frame buffer is needed.
local function render_delta(data, W, count)
    local unpack = table.unpack or unpack
    local pos = 1
    for _ = 1, count do
        local start = u16(data, pos)
        local len = data:byte(pos + 2)
        pos = pos + 3
        local x, y = start % W, floor(start / W)
        local text, fg, bg = {}, {}, {}
        for c = 1, len do
            text[c] = data:byte(pos)
            local col = data:byte(pos + 1)
            pos = pos + 2
            fg[c] = HEXB[col % 16]
            bg[c] = HEXB[floor(col / 16)]
        end
        if x + len <= W then             -- bound-check before drawing (spec §7)
            screen.setCursorPos(x + 1, y + 1)
            screen.blit(string.char(unpack(text)), string.char(unpack(fg)),
                        string.char(unpack(bg)))
        end
    end
end

-- Parse one video chunk payload into queued, absolutely-timestamped units.
-- Palette units take the PTS of the frame that follows them, so the due-drain
-- pops palette + keyframe together and flushes them in one redraw.
local function queue_video(chunk_pts, payload)
    local W, H = u16(payload, 1), u16(payload, 3)
    local n = W * H
    local raw_bytes = math.ceil(n / 8) * 5 + math.ceil(n / 2) * 2
    local pos, cur = 5, chunk_pts
    while pos <= #payload do
        local flags = payload:byte(pos)
        pos = pos + 1
        if flags < 128 then              -- palette unit: body is 48 bytes
            vqueue[#vqueue + 1] = { pts = cur, palette = payload:sub(pos, pos + 47) }
            pos = pos + 48
        else
            local enc = floor(flags / 16) % 8
            local duration = u16(payload, pos)
            pos = pos + 2
            if enc == 0 then             -- raw keyframe
                vqueue[#vqueue + 1] = { pts = cur, w = W, h = H,
                                        raw = payload:sub(pos, pos + raw_bytes - 1) }
                pos = pos + raw_bytes
            elseif enc == 1 then         -- delta: walk the spans to find its end
                local count = u16(payload, pos)
                local body = pos + 2
                pos = body
                for _ = 1, count do
                    pos = pos + 3 + payload:byte(pos + 2) * 2
                end
                vqueue[#vqueue + 1] = { pts = cur, w = W, count = count,
                                        delta = payload:sub(body, pos - 1) }
            elseif enc == 2 then         -- repeat: hold — nothing to draw
                -- no queue entry; the duration below still advances the clock
            else
                return                   -- unknown encoding: drop the rest of the GOP
            end
            cur = cur + duration
        end
    end
    while #vqueue > MAX_VIDEO_UNITS do
        table.remove(vqueue, 1)          -- runaway server: shed oldest
    end
    os.queueEvent("livecc_video")        -- wake the video player
end

-- After a forward clock jump, everything queued before `origin` is stale —
-- but deltas chain off their keyframe, so resume at the LAST raw keyframe at
-- or before origin (keeping the palette unit that precedes it) and let the
-- due-drain burst through the few remaining deltas up to "now".
local function prune_video(origin)
    local keep = nil
    for i = 1, #vqueue do
        local e = vqueue[i]
        if e.pts > origin then break end
        if e.raw then keep = i end
    end
    if not keep then return end
    if keep > 1 and vqueue[keep - 1].palette then keep = keep - 1 end
    if keep > 1 then
        local rest = {}
        for i = keep, #vqueue do rest[i - keep + 1] = vqueue[i] end
        vqueue = rest
    end
end

-- Render everything that is due.  All due units flush in ONE redraw: palettes
-- apply into the hidden window, so a palette+keyframe pair lands atomically.
local function drain_due()
    if #vqueue == 0 or anchor_pts == nil then return end
    local now = os.epoch("utc")
    if due_ms(vqueue[1].pts) > now then return end
    screen.setVisible(false)
    repeat
        local e = table.remove(vqueue, 1)
        if e.palette then
            apply_palette(e.palette)
        elseif e.raw then
            render_raw(e.raw, e.w, e.h)
        elseif e.delta then
            render_delta(e.delta, e.w, e.count)
        end
    until #vqueue == 0 or due_ms(vqueue[1].pts) > os.epoch("utc")
    screen.setVisible(true)          -- flush: palette + all cells land together
end

-- ── Anchoring ────────────────────────────────────────────────────────────────────

-- (Re)anchor the clock: `origin` plays at now + CLIENT_DELAY_MS.  On a real
-- jump (>1 s either way) buffered audio is flushed — the device would
-- otherwise play out samples timed against the OLD clock — and on a forward
-- jump stale video is pruned.  Both engines are then woken to re-evaluate.
local function set_anchor(origin)
    local old = clock_now()
    anchor_pts, anchor_ms = origin, os.epoch("utc") + CLIENT_DELAY_MS
    if old then
        local jump = clock_now() - old
        if jump > 48000 or jump < -48000 then
            flush_speakers()
        end
        if jump > 48000 then
            prune_video(origin)
        end
    end
    os.queueEvent("livecc_clock")
    os.queueEvent("livecc_video")
end

-- ── UI ──────────────────────────────────────────────────────────────────────────

local function mon_print(msg)
    -- Status text shares the buffered `screen` (the visible window covers the whole
    -- monitor, so writing straight to `mon` would sit behind it and be overwritten).
    screen.setBackgroundColor(colors.black)
    screen.setTextColor(colors.white)
    screen.clear()
    screen.setCursorPos(1, 1)
    screen.write(msg)
end

-- Human-readable server text (ERROR bodies) is printable ISO-8859-1 (spec §3):
-- the terminal font renders those bytes directly, so no decoding is needed.  Map
-- anything outside the printable ranges (0x20-0x7E, 0xA0-0xFF) to '?' so a non-
-- conforming server can't paint control glyphs or the 0x80-0x9F graphics blocks
-- across a message.
local function printable(s)
    return (s:gsub(".", function(ch)
        local b = ch:byte()
        if (b >= 0x20 and b <= 0x7E) or b >= 0xA0 then return ch end
        return "?"
    end))
end

-- --verbose: print every chunk's header (container header + payload header,
-- spec §4.1/§4.4/§4.6) as it's received.  Routed through console() like every
-- other status line, so it's silenced when the terminal itself is the display
-- (it would otherwise scribble over the video, same reasoning as elsewhere).
local COMPRESSION_NAMES = { [0] = "none", [1] = "deflate", [2] = "lz4", [3] = "zstd" }
local CODEC_NAMES = { [0] = "pcm8", [1] = "dfpwm" }

-- Walk a GOP's unit stream (spec §4.5) just to tally it for --verbose — same
-- traversal as queue_video, but nothing is queued/rendered here.
local function scan_video_units(payload, n)
    local raw_bytes = math.ceil(n / 8) * 5 + math.ceil(n / 2) * 2
    local pos = 5
    local palettes, raws, deltas, repeats, spans, duration = 0, 0, 0, 0, 0, 0
    while pos <= #payload do
        local flags = payload:byte(pos)
        pos = pos + 1
        if flags < 128 then                  -- palette unit: fixed 48-byte body
            palettes = palettes + 1
            pos = pos + 48
        else
            local enc = floor(flags / 16) % 8
            duration = duration + u16(payload, pos)
            pos = pos + 2
            if enc == 0 then                  -- raw keyframe
                raws = raws + 1
                pos = pos + raw_bytes
            elseif enc == 1 then               -- delta: walk spans to find its end
                deltas = deltas + 1
                local count = u16(payload, pos)
                spans = spans + count
                pos = pos + 2
                for _ = 1, count do
                    pos = pos + 3 + payload:byte(pos + 2) * 2
                end
            elseif enc == 2 then                -- repeat: no body
                repeats = repeats + 1
            else
                break                            -- unknown encoding: stop the tally
            end
        end
    end
    return palettes, raws, deltas, repeats, spans, duration
end

local function verbose_chunk(pts, length, ctype, compression, payload)
    local seconds = pts / 48000
    if ctype == TYPE_VIDEO then
        local w, h = u16(payload, 1), u16(payload, 3)
        local line = string.format(
            "LiveCC: [video] pts=%d (%.2fs) len=%d %dx%d compression=%s",
            pts, seconds, length, w, h, COMPRESSION_NAMES[compression] or tostring(compression))
        if compression == 0 then                     -- only "none" is decodable (spec §4.1.2)
            local palettes, raws, deltas, repeats, spans, duration =
                scan_video_units(payload, w * h)
            line = line .. string.format(
                " units=%d(palette)+%d(raw)+%d(delta,%d spans)+%d(repeat) gop_dur=%.2fs",
                palettes, raws, deltas, spans, repeats, duration / 48000)
        end
        console(line)
    elseif ctype == TYPE_AUDIO then
        local hdr = payload:byte(1)
        local codec, role = floor(hdr / 16), hdr % 16
        local samples = codec == 1 and (#payload - 1) * 8 or (#payload - 1)
        console(string.format(
            "LiveCC: [audio] pts=%d (%.2fs) len=%d codec=%s role=%s samples=%d",
            pts, seconds, length, CODEC_NAMES[codec] or tostring(codec),
            ROLE_NAMES[role] or tostring(role), samples))
    else
        console(string.format("LiveCC: [type %d] pts=%d (%.2fs) len=%d", ctype, pts, seconds, length))
    end
end

local errored = false   -- an ERROR was shown; don't overwrite it with "Stream ended"
local ended = false

local function handle_control(opcode, body)
    if opcode == OP_STATUS then
        local state = body:byte(1)
        if state == 0 then
            mon_print("Buffering...")
        elseif state == 2 then
            ended = true                         -- terminal: stream complete (spec §5.6)
        else
            -- playing: anchor to the origin PTS when present (spec §5.6); a
            -- draft-00 server sends none — first-chunk fallback covers that.
            if #body >= 7 then
                set_anchor(u48(body, 2))
            end
            if not WANT_VIDEO then
                mon_print("Playing (audio only)")   -- no frames paint over this
            end
            -- else the next due video frame paints over "Buffering..."
        end
    elseif opcode == OP_ERROR then
        errored = true
        local text = printable(body)
        console("LiveCC error: " .. text)
        mon_print("Error: " .. text)
    end
end

-- Dispatch one binary message: a marker-led media chunk, or a control frame.
local function handle_message(msg)
    if msg:byte(1) == MARKER then
        -- chunk: [marker][pts u48][length u24][type u8][compression u8][payload]
        local pts = u48(msg, 2)
        local length = u24(msg, 8)
        local ctype = msg:byte(11)
        local compression = msg:byte(12)
        local payload = msg:sub(13)
        if VERBOSE then verbose_chunk(pts, length, ctype, compression, payload) end
        if compression ~= 0 then return end                    -- compressed payloads deferred (CC = none)
        if ctype == TYPE_VIDEO and WANT_VIDEO then
            if not anchor_pts then set_anchor(pts) end   -- draft-00 fallback
            queue_video(pts, payload)
        elseif ctype == TYPE_AUDIO and WANT_AUDIO then
            if not anchor_pts then set_anchor(pts) end   -- draft-00 fallback
            handle_audio(pts, payload)
        end
        -- unknown chunk types: skip (self-describing stream, spec §6)
    else
        local opcode = msg:byte(1)
        handle_control(opcode, msg:sub(4, 3 + u16(msg, 2)))
    end
end

-- ── Connect + handshake ─────────────────────────────────────────────────────────

console("LiveCC: connecting to server...")
mon_print("Connecting to server...")

local WS_BASE = SERVER:gsub("^http", "ws")
local ok, err = http.websocket(WS_BASE .. "/ws/play")
if not ok then
    die("Connection failed: " .. (err or "unknown error"))
end
ws = ok

-- Lock-step handshake (spec §5.3): each step is answered by ACK, or by ERROR
-- with a human-readable reason (e.g. a sync room settings mismatch).
local function expect_ack(step)
    while true do
        local recv_ok, msg, is_binary = pcall(ws.receive)
        if not recv_ok or msg == nil then
            die("Server closed the connection during " .. step .. ".")
        end
        if is_binary and msg:byte(1) ~= MARKER then
            local opcode = msg:byte(1)
            local body = msg:sub(4, 3 + u16(msg, 2))
            if opcode == OP_ACK then return end
            if opcode == OP_ERROR then
                ws.close()
                die("Server refused " .. step .. ": " .. printable(body))
            end
        end
    end
end

ws.send(control_frame(OP_ROOM, room_body()), true)
expect_ack("ROOM")
ws.send(control_frame(OP_CAPS, caps_body()), true)
expect_ack("CAPS")
ws.send(control_frame(OP_START), true)

-- ── Main loop ─────────────────────────────────────────────────────────────────

console("LiveCC: streaming \7 press Q to quit")   -- \7 = CC bullet dingbat (no em-dash in the font)

-- Build the coroutine set.  parallel.waitForAny stops as soon as ANY function
-- returns, so the audio engine must only join when audio is actually on — adding
-- a no-op coroutine that returns immediately would tear the whole player down
-- before the receiver rendered anything (the old no-speaker "stuck on Connecting"
-- bug).
local tasks = {}

-- Receiver: parse chunks/controls, queue video units and audio, handle status
tasks[#tasks + 1] = function()
    while not ended do
        -- pcall: once the server closes the socket (after an ERROR, or at stream
        -- end), ws.receive() raises "attempt to use a closed file" instead of
        -- returning nil — so a bare call would crash the program right after the
        -- error was shown.  Treat any failure or nil as "stop cleanly".
        local recv_ok, msg, is_binary = pcall(ws.receive)
        if not recv_ok or msg == nil then break end
        if is_binary then
            handle_message(msg)
        end
    end
    -- Let queued video AND the audio tail play out before declaring the
    -- stream over (bounded: a missing clock must not hang shutdown).
    local deadline = os.epoch("utc") + 10000
    while not errored and os.epoch("utc") < deadline
          and (#vqueue > 0 or audio_pending()) do
        local t = os.startTimer(0.25)
        repeat
            local ev, id = os.pullEvent()
        until ev == "timer" and id == t
    end
    if not errored then mon_print("Stream ended.") end   -- keep any error on screen
    console("LiveCC: stream ended.")
end

-- Video player: renders queued units at their PTS against the media clock.
if WANT_VIDEO then
    tasks[#tasks + 1] = function()
        while true do
            drain_due()
            if #vqueue > 0 and anchor_pts then
                local wait = (due_ms(vqueue[1].pts) - os.epoch("utc")) / 1000
                if wait > 0 then
                    local timer = os.startTimer(wait)
                    repeat
                        local ev, id = os.pullEvent()
                    until (ev == "timer" and id == timer)
                        or ev == "livecc_video" or ev == "livecc_clock"
                end
            else
                repeat
                    local ev = os.pullEvent()
                until ev == "livecc_video" or ev == "livecc_clock"
            end
        end
    end
end

-- Audio engine: ONE task drives every speaker of every role against the
-- clock (see the audio section above).  Only present when audio is on
-- (WANT_AUDIO already requires at least one speaker).
if WANT_AUDIO then
    tasks[#tasks + 1] = function()
        while true do
            local wait_ms = feed_roles()
            local timer = wait_ms and os.startTimer(math.max(wait_ms, 5) / 1000)
            repeat
                local ev, id = os.pullEvent()
            until ev == "livecc_audio" or ev == "livecc_clock"
                or ev == "speaker_audio_empty"
                or (timer and ev == "timer" and id == timer)
        end
    end
end

-- Key watcher
tasks[#tasks + 1] = function()
    while true do
        local _, key = os.pullEvent("key")
        if key == keys.q then
            if ws then
                pcall(function() ws.send(control_frame(OP_QUIT), true) end)
                ws.close()
            end
            flush_speakers()             -- don't let buffered audio play on
            mon_print("Stopped.")
            console("LiveCC: stopped by user.")
            return
        end
    end
end

-- Test hook: the server repo's Lua suite (server/tests/test_player_lua.py +
-- cc_stubs.lua) loads this script with _LIVECC_TEST set and every CC API
-- stubbed, then drives these internals directly instead of entering the event
-- loop.  Production CC never defines _LIVECC_TEST, so this block is inert
-- there.  Keep additions here in sync with what the suite actually covers.
if _LIVECC_TEST then
    return {
        -- media clock
        clock_now = clock_now,
        set_anchor = set_anchor,
        get_anchor = function() return anchor_pts, anchor_ms end,
        -- wire handling
        handle_message = handle_message,
        -- audio engine
        feed_roles = feed_roles,
        audio_pending = audio_pending,
        flush_speakers = flush_speakers,
        audio_queues = audio_queues,
        role_state = role_state,
        speakers_by_role = speakers_by_role,
        -- video engine (vqueue is reassigned on prune, so expose a getter)
        drain_due = drain_due,
        get_vqueue = function() return vqueue end,
        -- negotiated state
        channels_mask = channels_mask,
        is_ended = function() return ended end,
        wants = function() return WANT_VIDEO, WANT_AUDIO end,
    }
end

parallel.waitForAny(table.unpack(tasks))
