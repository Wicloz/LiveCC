-- LiveCC player  —  watches YouTube videos / live streams on a CC monitor (or the
--                   computer's own terminal when none is attached, e.g. a tablet)
--                   plays audio on ALL attached speakers simultaneously
-- Run `livecc --help` for usage.
--
-- One WebSocket carries everything.  Binary messages are tagged by a leading
-- opcode byte: 0x01 = video frame (a sanjuuni 32vid uncompressed frame), 0x02 =
-- audio (raw 8-bit PCM).  Text messages are status: "META w h fps", "BUFFERING",
-- "PLAYING", "ERROR ...".

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
    print("  --crunchy     Low-bandwidth mode: lower resolution + 1-bit audio")
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
local START_TS, END_TS, LOOP, HELP, CRUNCHY, SYNC = nil, nil, false, false, false, false
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
        elseif a == "--crunchy" then
            CRUNCHY = true
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

-- Larger text scale = bigger glyphs = fewer cells = lower resolution.  --crunchy
-- uses 1.0 (~1/4 the cells of the default 0.5), cutting video bandwidth.  Monitor
-- only — a terminal has a fixed size, so there --crunchy just affects audio.
if mon.setTextScale then mon.setTextScale(CRUNCHY and 1.0 or 0.5) end
mon.setCursorBlink(false)            -- no blinking cursor in the corner
local TERM_W, TERM_H = mon.getSize()

-- Draw everything through a buffered window so each frame's palette change AND its
-- cell blits reach the monitor in ONE redraw (setVisible(false) -> draw ->
-- setVisible(true) flushes the window's palette and contents together).  The
-- adaptive per-frame palette calls setPaletteColour for all 16 entries every frame;
-- applied straight to the monitor that instantly recolours the PREVIOUS frame still
-- on screen, so if a monitor refresh lands between the palette change and the
-- row-by-row blit a ghost of the old frame flashes in the new frame's palette (often
-- near-monochrome or inverted) — the flicker/tearing.  Buffering makes it atomic.
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

local ws = nil

-- ── Audio: DFPWM decoded natively, drained by its own coroutine ─────────────────
-- Decoding is fast (C-backed) so it never hitches video.  A bounded queue keeps
-- back-pressure from blocking rendering; drop oldest if the client falls behind.

local audio_queue = {}
local MAX_AUDIO_CHUNKS = 80            -- ~8 s at 0.1 s/chunk (safety cap)

-- Audio decode depends on the codec the server is sending (chosen by --crunchy):
--   default  raw unsigned 8-bit PCM — the speaker's native format; just unpack
--            each byte to a signed amplitude (byte - 128 gives -128..127), no
--            decode state.  Sliced to stay under string.byte's multi-return limit.
--   crunchy  1-bit DFPWM — lower bandwidth, but needs the stateful CC decoder.
-- Either way chunks are short (~0.1 s) so this stays brief and interleaves with
-- video rendering instead of stalling it.
local decode_audio
if CRUNCHY then
    local decoder = require("cc.audio.dfpwm").make_decoder()
    decode_audio = function(data) return decoder(data) end
else
    decode_audio = function(data)
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

-- ── 32vid stream (sanjuuni format) ───────────────────────────────────────────────
-- The binary side of the socket is a 32vid stream: a 12-byte "32VD" header
-- (W, H, fps, nstreams, flags) then self-delimiting chunks (<size, datalength,
-- type> + data) interleaving video (type 0) and audio (type 1).
-- An uncompressed video frame (compression mode 0), W x H cells, is:
--   screen : ceil(W*H/8)*5 bytes — 8 cells packed into 5 bytes, MSB-first; each
--            cell is the 5-bit drawing-char index (the blit char is 0x80 + index)
--   colour : W*H bytes — one per cell, (bg << 4) | fg  (palette indices 0-15)
--   palette: 48 bytes — 16 * (R,G,B), applied via setPaletteColour (only on change)
-- Lua 5.1 has no bitwise operators, so the 5-bit codes are unpacked with arithmetic.

local VID_W, VID_H = TERM_W, TERM_H      -- grid; overwritten by the 32vid header
local HEXB = {}                          -- palette index 0-15 -> blit hex char byte
for i = 0, 15 do HEXB[i] = ("0123456789abcdef"):byte(i + 1) end
local last_palette = nil                 -- last applied palette block (skip no-ops)

local function render_frame(data, W, H)
    local n = W * H
    local floor = math.floor
    local unpack = table.unpack or unpack

    -- 1. screen: unpack 8 five-bit codes from every 5 bytes into char bytes.
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

    -- 2. colour: one byte per cell, bg in the high nibble, fg in the low nibble.
    local fg, bg = {}, {}
    for i = 1, n do
        local cb = data:byte(p); p = p + 1
        fg[i] = HEXB[cb % 16]
        bg[i] = HEXB[floor(cb / 16)]
    end

    -- Buffer the palette change and the blits, then flush them in one redraw so the
    -- monitor never shows the old frame recoloured by the new palette (see `screen`).
    screen.setVisible(false)

    -- 3. palette: apply only when it differs from the last frame's (a no-op for a
    --    fixed palette; changes every frame for the adaptive per-frame palette).
    local palette = data:sub(p, p + 47)
    if palette ~= last_palette then
        if screen.setPaletteColour then
            for i = 0, 15 do
                local o = p + i * 3
                screen.setPaletteColour(2 ^ i, data:byte(o) / 255,
                    data:byte(o + 1) / 255, data:byte(o + 2) / 255)
            end
        end
        last_palette = palette
    end

    -- 4. blit row by row.
    for y = 1, H do
        local base = (y - 1) * W
        screen.setCursorPos(1, y)
        screen.blit(string.char(unpack(text, base + 1, base + W)),
                    string.char(unpack(fg, base + 1, base + W)),
                    string.char(unpack(bg, base + 1, base + W)))
    end

    screen.setVisible(true)          -- flush: palette + all cells land together
end

-- Bytes one uncompressed frame occupies in a video chunk.
local function frame_bytes(W, H)
    return math.ceil(W * H / 8) * 5 + W * H + 48
end

-- Parse the 12-byte "32VD" file header; updates the grid the decoder renders to.
local function read_header(msg)
    VID_W = msg:byte(5) + msg:byte(6) * 256
    VID_H = msg:byte(7) + msg:byte(8) * 256
    -- byte 9 = fps, 10 = nstreams, 11-12 = flags; the client paces off arrival, so
    -- it only needs the grid (audio codec is already known from --crunchy).
    console("LiveCC: connected")
end

-- Dispatch one binary 32vid message: the file header, or a typed chunk.
local function handle_binary(msg)
    if msg:sub(1, 4) == "32VD" then
        read_header(msg)
        return
    end
    -- chunk: <size u32><datalength u32><type u8> + data  (all little-endian)
    local datalen = msg:byte(5) + msg:byte(6) * 256 + msg:byte(7) * 65536 + msg:byte(8) * 16777216
    local ctype = msg:byte(9)
    if ctype == 0 then                       -- video: datalen frames back to back
        local fsz = frame_bytes(VID_W, VID_H)
        local pos = 10
        for _ = 1, datalen do
            render_frame(msg:sub(pos, pos + fsz - 1), VID_W, VID_H)
            pos = pos + fsz
        end
    elseif ctype == 1 and WANT_AUDIO then    -- audio: PCM, or DFPWM if --crunchy
        audio_queue[#audio_queue + 1] = decode_audio(msg:sub(10))
        while #audio_queue > MAX_AUDIO_CHUNKS do
            table.remove(audio_queue, 1)     -- drop oldest, stay in sync
        end
        os.queueEvent("livecc_audio")        -- wake the audio player
    end
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

local function handle_text(msg)
    if msg == "BUFFERING" then
        mon_print("Buffering...")
    elseif msg == "PLAYING" then
        if not WANT_VIDEO then
            mon_print("Playing (audio only)")   -- no frames will paint over this
        end
        -- else the next video frame paints over the "Buffering..." message
    elseif msg:sub(1, 5) == "ERROR" then
        errored = true
        local detail = msg:sub(7)
        console("LiveCC error: " .. detail)
        mon_print("Error: " .. detail)
    end
end

-- ── Connect ───────────────────────────────────────────────────────────────────

console("LiveCC: connecting to server...")
mon_print("Connecting to server...")

local WS_BASE = SERVER:gsub("^http", "ws")
local url = WS_BASE
    .. "/ws/play"
    .. "?url="    .. textutils.urlEncode(VIDEO_URL)
    .. "&width="  .. tostring(TERM_W)
    .. "&height=" .. tostring(TERM_H)
    .. "&fps="    .. tostring(WANTED_FPS)
    .. "&audio="  .. (WANT_AUDIO and "1" or "0")
    .. "&video="  .. (WANT_VIDEO and "1" or "0")
    .. (START_TS and ("&start=" .. textutils.urlEncode(START_TS)) or "")
    .. (END_TS   and ("&end="   .. textutils.urlEncode(END_TS))   or "")
    .. (LOOP and "&loop=1" or "")
    .. (SYNC and "&sync=1" or "")
    .. (CRUNCHY and "&crunchy=1" or "")

local ok, err = http.websocket(url)
if not ok then
    die("Connection failed: " .. (err or "unknown error"))
end
ws = ok

-- ── Main loop ─────────────────────────────────────────────────────────────────

console("LiveCC: streaming — press Q to quit")

-- Build the coroutine set.  parallel.waitForAny stops as soon as ANY function
-- returns, so the audio player must only join when audio is actually on — adding
-- a no-op coroutine that returns immediately would tear the whole player down
-- before the receiver rendered anything (the old no-speaker "stuck on Connecting"
-- bug).
local tasks = {}

-- Receiver: render video, enqueue audio, handle status
tasks[#tasks + 1] = function()
    while true do
        -- pcall: once the server closes the socket (after an ERROR, or at stream
        -- end), ws.receive() raises "attempt to use a closed file" instead of
        -- returning nil — so a bare call would crash the program right after the
        -- error was shown.  Treat any failure or nil as "stop cleanly".
        local ok, msg, is_binary = pcall(ws.receive)
        if not ok or msg == nil then break end
        if is_binary then
            handle_binary(msg)      -- 32vid header or a video/audio chunk
        else
            handle_text(msg)        -- BUFFERING / PLAYING / ERROR
        end
    end
    if not errored then mon_print("Stream ended.") end   -- keep any error on screen
    console("LiveCC: stream ended.")
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
            if ws then ws.close() end
            mon_print("Stopped.")
            console("LiveCC: stopped by user.")
            return
        end
    end
end

parallel.waitForAny(table.unpack(tasks))
