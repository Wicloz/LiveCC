-- LiveCC player  —  watches YouTube videos / live streams on a CC monitor (or the
--                   computer's own terminal when none is attached, e.g. a tablet)
--                   plays audio on ALL attached speakers simultaneously
-- Run `livecc --help` for usage.
--
-- One WebSocket carries everything, speaking CCMF (the ComputerCraft Media
-- Format, docs/cc-media-format.md).  Every message is binary: a message whose
-- first byte is the container marker (67, "C") is a media chunk — a video GOP
-- (palette + raw/delta/repeat frame units) or ~0.1 s of audio — and anything
-- else is a control frame [opcode][len u16][body]: STATUS, ERROR, END.
-- The client opens with ROOM -> ACK, CAPS -> ACK, START, then just consumes.

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
    print("  -h, --help    Show this help and exit")
    print("")
    print("Timestamps (TS): 90, 90s, 1m30s, 3h2m  (default unit: seconds)")
    print("")
    print("Examples:")
    print("  livecc https://youtu.be/dQw4w9WgXcQ")
    print("  livecc https://youtu.be/XYZ --fps 30")
    print("  livecc https://youtu.be/XYZ --start 1m30s --loop")
end

-- Parse: one positional (url) plus optional flags.
local positional = {}
local START_TS, END_TS, LOOP, HELP, SYNC = nil, nil, false, false, false
local NO_AUDIO, NO_VIDEO = false, false
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

-- One physical speaker can be reachable under two names (e.g. adjacent AND via
-- a wired modem).  Driving it twice plays every audio segment twice.  Two wraps
-- of the same speaker share one audio buffer, so we detect duplicates: fill one
-- buffer with silence and check whether the other then reports full.
local function dedupe_speakers(found)
    if #found < 2 then return found end
    local silence = {}
    for i = 1, 8192 do silence[i] = 0 end
    local unique = {}
    for _, s in ipairs(found) do
        local dup = false
        for _, u in ipairs(unique) do
            u.stop(); s.stop()
            local guard = 0
            while u.playAudio(silence) and guard < 64 do guard = guard + 1 end
            if not s.playAudio(silence) then dup = true end   -- shares u's buffer
            u.stop(); s.stop()
            if dup then break end
        end
        if not dup then unique[#unique + 1] = s end
    end
    for _, u in ipairs(unique) do u.stop() end
    return unique
end

local found_speakers = {}
for _, name in ipairs(peripheral.getNames()) do
    if peripheral.getType(name) == "speaker" then
        found_speakers[#found_speakers + 1] = peripheral.wrap(name)
    end
end
local speakers = dedupe_speakers(found_speakers)

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
    console(msg)
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
local OP_ACK, OP_ERROR, OP_STATUS, OP_END = 5, 6, 7, 8

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

local function caps_body()
    local flags = (WANT_VIDEO and 1 or 0) + (WANT_AUDIO and 2 or 0)
    local fps = math.max(1, math.min(255, floor(WANTED_FPS)))
    return string.char(flags)
        .. string.char(0)                -- video: no optional encodings yet
        .. string.char(3)                -- audio: pcm8 + dfpwm both decodable
        .. pack_u16(1)                   -- channels: mono only
        .. string.char(1)                -- compression: none only
        .. pack_u16(TERM_W) .. pack_u16(TERM_H) .. string.char(fps)
end

local ws = nil

-- ── Audio ────────────────────────────────────────────────────────────────────────
-- The codec arrives per chunk in the a-hdr nibble, matching what we advertised:
--   pcm8   raw unsigned 8-bit PCM — the speaker's native format; just unpack
--          each byte to a signed amplitude (byte - 128 gives -128..127), no
--          decode state.  Sliced to stay under string.byte's multi-return limit.
--   dfpwm  1-bit DFPWM — needs the CC decoder; its state MUST reset per chunk
--          (spec §4.6), so each chunk gets a fresh decoder.
-- Chunks are short (~0.1 s) so decoding stays brief and interleaves with video
-- rendering instead of stalling it.  A bounded queue keeps back-pressure from
-- blocking rendering; drop oldest if the client falls behind.

local audio_queue = {}
local MAX_AUDIO_CHUNKS = 80            -- ~8 s at 0.1 s/chunk (safety cap)

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

local function handle_audio(payload)
    local hdr = payload:byte(1)
    local codec, channel = floor(hdr / 16), hdr % 16
    if channel ~= 0 then return end      -- we only asked for mono
    local samples
    if codec == 0 then
        samples = decode_pcm8(payload:sub(2))
    elseif codec == 1 then
        samples = dfpwm.make_decoder()(payload:sub(2))
    else
        return                           -- unknown codec: skip the chunk
    end
    audio_queue[#audio_queue + 1] = samples
    while #audio_queue > MAX_AUDIO_CHUNKS do
        table.remove(audio_queue, 1)     -- drop oldest, stay in sync
    end
    os.queueEvent("livecc_audio")        -- wake the audio player
end

local function play_on_all(samples)
    if #samples == 0 then return end
    local pending = {}
    for i, spk in ipairs(speakers) do
        if not spk.playAudio(samples) then pending[i] = true end
    end
    while next(pending) do
        os.pullEvent("speaker_audio_empty")
        for i in pairs(pending) do
            if speakers[i].playAudio(samples) then pending[i] = nil end
        end
    end
end

-- ── Video: unit queue + PTS pacing ──────────────────────────────────────────────
-- A video chunk is a self-contained GOP: [w u16][h u16][compression u8] then a
-- unit stream — a 48-byte palette unit, a raw keyframe, and delta/repeat units,
-- each frame carrying its hold duration in 48 kHz samples.  The server releases
-- whole GOPs slightly early, so the client itself schedules each frame:
-- arriving units land in `vqueue` with an absolute PTS, and the player coroutine
-- renders whatever is due against a media clock (os.epoch is real time, ms).
--
-- The clock anchors on the first chunk (plus a little startup slack so the queue
-- never starts empty).  Chunks arriving LATE re-anchor after 1 s (server
-- re-buffer: resume smoothly); chunks arriving EARLY are simply queued — only a
-- big PTS jump (live-edge skip) re-anchors, dropping the stale queue.

local vqueue = {}
local anchor_pts, anchor_ms = nil, nil
local START_SLACK_MS = 200
-- Re-anchor thresholds are asymmetric.  LATE (chunk arrives after its due
-- time): the server stalled/re-buffered — resume promptly, so 1 s.  EARLY
-- (chunk PTS is far ahead of the clock): normal operation is ALWAYS somewhat
-- early — the server releases video ahead of the clock and live-edge skips
-- add more — so small earliness must just queue (re-anchoring on it burst-
-- rendered the whole queue and then froze until the next chunk).  Only a big
-- jump (a genuine live-edge skip) re-anchors, and then the stale queue is
-- DROPPED, not burst-rendered.
local RESYNC_LATE_MS = 1000
local RESYNC_EARLY_MS = 3000

local function due_ms(pts)
    return anchor_ms + (pts - anchor_pts) / 48   -- 48 samples per ms
end

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
    local compression = payload:byte(5)
    if compression ~= 0 then return end  -- only "none" is decodable (deferred)
    local n = W * H
    local raw_bytes = math.ceil(n / 8) * 5 + math.ceil(n / 2) * 2
    local pos, cur = 6, chunk_pts
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
    os.queueEvent("livecc_video")        -- wake the video player
end

-- Anchor (or re-anchor) the media clock from an arriving chunk's PTS.
local function sync_clock(pts)
    local now = os.epoch("utc")
    if anchor_pts == nil then
        anchor_pts, anchor_ms = pts, now + START_SLACK_MS
        return
    end
    local lateness = now - due_ms(pts)
    if lateness > RESYNC_LATE_MS then
        -- Server stalled/re-buffered: shift the clock so playback resumes
        -- smoothly (the queue is empty or overdue anyway).
        anchor_pts, anchor_ms = pts, now + START_SLACK_MS
    elseif lateness < -RESYNC_EARLY_MS then
        -- Live-edge skip: the stream jumped ahead.  Jump with it — drop
        -- whatever is still queued (it is stale content behind the edge)
        -- rather than fast-forwarding through it on screen.
        vqueue = {}
        anchor_pts, anchor_ms = pts, now + START_SLACK_MS
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

local errored = false   -- an ERROR was shown; don't overwrite it with "Stream ended"
local ended = false

local function handle_control(opcode, body)
    if opcode == OP_STATUS then
        if body:byte(1) == 0 then
            mon_print("Buffering...")
        elseif not WANT_VIDEO then
            mon_print("Playing (audio only)")   -- no frames will paint over this
        end
        -- else the next video frame paints over the "Buffering..." message
    elseif opcode == OP_ERROR then
        errored = true
        console("LiveCC error: " .. body)
        mon_print("Error: " .. body)
    elseif opcode == OP_END then
        ended = true
    end
end

-- Dispatch one binary message: a marker-led media chunk, or a control frame.
local function handle_message(msg)
    if msg:byte(1) == MARKER then
        -- chunk: [marker][pts u48][length u24][type u8][payload]
        local pts = u48(msg, 2)
        local ctype = msg:byte(11)
        if ctype == TYPE_VIDEO and WANT_VIDEO then
            sync_clock(pts)
            queue_video(pts, msg:sub(12))
        elseif ctype == TYPE_AUDIO and WANT_AUDIO then
            handle_audio(msg:sub(12))
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
                die("Server refused " .. step .. ": " .. body)
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

console("LiveCC: streaming — press Q to quit")

-- Build the coroutine set.  parallel.waitForAny stops as soon as ANY function
-- returns, so the audio player must only join when audio is actually on — adding
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
    -- Let queued video play out before declaring the stream over.
    while #vqueue > 0 and not errored do
        os.pullEvent("livecc_drained")
    end
    if not errored then mon_print("Stream ended.") end   -- keep any error on screen
    console("LiveCC: stream ended.")
end

-- Video player: renders queued units at their PTS against the media clock.
if WANT_VIDEO then
    tasks[#tasks + 1] = function()
        while true do
            drain_due()
            if #vqueue > 0 then
                local wait = (due_ms(vqueue[1].pts) - os.epoch("utc")) / 1000
                if wait > 0 then
                    local timer = os.startTimer(wait)
                    repeat
                        local ev, id = os.pullEvent()
                    until (ev == "timer" and id == timer) or ev == "livecc_video"
                end
            else
                os.queueEvent("livecc_drained")   -- receiver may be waiting to end
                os.pullEvent("livecc_video")
            end
        end
    end
end

-- Audio player: drains the queue; may block on speaker_audio_empty.  Only present
-- when audio is on (WANT_AUDIO already requires at least one speaker).
if WANT_AUDIO then
    tasks[#tasks + 1] = function()
        while true do
            if #audio_queue > 0 then
                play_on_all(table.remove(audio_queue, 1))
            else
                os.pullEvent("livecc_audio")   -- sleep until a chunk arrives
            end
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
            mon_print("Stopped.")
            console("LiveCC: stopped by user.")
            return
        end
    end
end

parallel.waitForAny(table.unpack(tasks))
