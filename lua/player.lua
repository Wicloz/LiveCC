-- LiveCC player  —  watches YouTube videos / live streams on a CC monitor (or the
--                   computer's own terminal when none is attached, e.g. a tablet)
--                   plays audio on ALL attached speakers simultaneously
-- Run `livecc --help` for usage.
--
-- One WebSocket carries everything.  Binary messages are tagged by a leading
-- opcode byte: 0x01 = video frame, 0x02 = audio (raw 8-bit PCM).  Text messages are
-- status: "META w h fps", "BUFFERING", "PLAYING", "ERROR ...".

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

-- ── Video rendering (binary frame) ──────────────────────────────────────────────
-- bytes 1-2 = W (uint16 BE), 3-4 = H, then H rows of (W text, W fg, W bg).

local function render_frame(data)
    local W = data:byte(1) * 256 + data:byte(2)
    local H = data:byte(3) * 256 + data:byte(4)
    local row_len = W * 3
    local pos = 5
    for y = 1, H do
        local text = data:sub(pos,         pos + W - 1)
        local fg   = data:sub(pos + W,     pos + 2 * W - 1)
        local bg   = data:sub(pos + 2 * W, pos + 3 * W - 1)
        mon.setCursorPos(1, y)
        mon.blit(text, fg, bg)
        pos = pos + row_len
    end
end

-- ── UI ──────────────────────────────────────────────────────────────────────────

local function mon_print(msg)
    mon.setBackgroundColor(colors.black)
    mon.setTextColor(colors.white)
    mon.clear()
    mon.setCursorPos(1, 1)
    mon.write(msg)
end

local errored = false   -- an ERROR was shown; don't overwrite it with "Stream ended"

local function handle_text(msg)
    if msg:sub(1, 4) == "META" then
        console("LiveCC: connected")
        mon_print("Buffering...")
    elseif msg == "BUFFERING" then
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
            local op = msg:byte(1)
            if op == 1 then
                render_frame(msg:sub(2))
            elseif op == 2 and WANT_AUDIO then
                -- Decode/unpack audio (codec per --crunchy) and enqueue, in
                -- arrival order to keep the DFPWM decoder's state in sync.
                audio_queue[#audio_queue + 1] = decode_audio(msg:sub(2))
                while #audio_queue > MAX_AUDIO_CHUNKS do
                    table.remove(audio_queue, 1)   -- drop oldest, stay in sync
                end
                os.queueEvent("livecc_audio")      -- wake the audio player
            end
        else
            handle_text(msg)
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
