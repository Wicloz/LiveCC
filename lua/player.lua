-- LiveCC player  —  watches YouTube videos / live streams on CC monitors
--                   plays audio on ALL attached speakers simultaneously
-- Run `livecc --help` for usage.
--
-- One WebSocket carries everything.  Binary messages are tagged by a leading
-- opcode byte: 0x01 = video frame, 0x02 = audio (DFPWM).  Text messages are
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
    print("livecc - play videos and live streams on a ComputerCraft monitor")
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
local START_TS, END_TS, LOOP, HELP = nil, nil, false, false
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

-- ── Monitor + speakers ─────────────────────────────────────────────────────────

local mon = peripheral.find("monitor")
if not mon then die("No monitor found — attach a monitor and try again.") end
mon.setTextScale(0.5)
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
if #speakers == 0 then
    print("LiveCC: no speakers found - audio disabled")
else
    local msg = "LiveCC: " .. #speakers .. " speaker(s)"
    if #found_speakers > #speakers then
        msg = msg .. " (" .. (#found_speakers - #speakers) .. " duplicate ignored)"
    end
    print(msg)
end

local ws = nil

-- ── Audio: DFPWM decoded natively, drained by its own coroutine ─────────────────
-- Decoding is fast (C-backed) so it never hitches video.  A bounded queue keeps
-- back-pressure from blocking rendering; drop oldest if the client falls behind.

local dfpwm = require("cc.audio.dfpwm")
local decoder = dfpwm.make_decoder()   -- stateful: feed chunks in arrival order
local audio_queue = {}
local MAX_AUDIO_CHUNKS = 8             -- ~8 s at 1 s/chunk (safety cap)

-- DFPWM postfilter.  A raw 1-bit decode carries a ~Nyquist/2 whine that's
-- intrinsic to the codec; cc.audio.dfpwm returns those samples unfiltered.  The
-- reference spec removes it with a one-pole low-pass on the decoded amplitudes:
--   out += (sample - out) * (POSTFILTER_K / 256)
-- Lower K = smoother (less whine, duller); 100/256 is the spec's suggestion.
-- State (out) persists across chunks so the filter stays continuous, so — like
-- the decoder — chunks must be filtered in arrival order.
local POSTFILTER_K = 100
local _pf_out = 0

local function postfilter(samples)
    for i = 1, #samples do
        _pf_out = _pf_out + (samples[i] - _pf_out) * POSTFILTER_K / 256
        local v = math.floor(_pf_out + 0.5)
        if v > 127 then v = 127 elseif v < -128 then v = -128 end
        samples[i] = v
    end
    return samples
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

local function handle_text(msg)
    if msg:sub(1, 4) == "META" then
        print("LiveCC: connected")
        mon_print("Buffering...")
    elseif msg == "BUFFERING" then
        mon_print("Buffering...")
    elseif msg == "PLAYING" then
        -- next frame will paint over this
    elseif msg:sub(1, 5) == "ERROR" then
        local detail = msg:sub(7)
        print("LiveCC error: " .. detail)
        mon_print("Error: " .. detail)
    end
end

-- ── Connect ───────────────────────────────────────────────────────────────────

print("LiveCC: connecting to server...")
mon_print("Connecting to server...")

local WS_BASE = SERVER:gsub("^http", "ws")
local url = WS_BASE
    .. "/ws/play"
    .. "?url="    .. textutils.urlEncode(VIDEO_URL)
    .. "&width="  .. tostring(TERM_W)
    .. "&height=" .. tostring(TERM_H)
    .. "&fps="    .. tostring(WANTED_FPS)
    .. "&audio="  .. (#speakers > 0 and "1" or "0")
    .. (START_TS and ("&start=" .. textutils.urlEncode(START_TS)) or "")
    .. (END_TS   and ("&end="   .. textutils.urlEncode(END_TS))   or "")
    .. (LOOP and "&loop=1" or "")

local ok, err = http.websocket(url)
if not ok then
    die("Connection failed: " .. (err or "unknown error"))
end
ws = ok

-- ── Main loop ─────────────────────────────────────────────────────────────────

print("LiveCC: streaming — press Q to quit")

parallel.waitForAny(

    -- Receiver: render video, enqueue audio, handle status
    function()
        while true do
            local msg, is_binary = ws.receive()
            if msg == nil then break end
            if is_binary then
                local op = msg:byte(1)
                if op == 1 then
                    render_frame(msg:sub(2))
                elseif op == 2 then
                    -- Decode + postfilter here, in arrival order, to keep both
                    -- the decoder and the postfilter state in sync.
                    audio_queue[#audio_queue + 1] = postfilter(decoder(msg:sub(2)))
                    while #audio_queue > MAX_AUDIO_CHUNKS do
                        table.remove(audio_queue, 1)   -- drop oldest, stay in sync
                    end
                    os.queueEvent("livecc_audio")      -- wake the audio player
                end
            else
                handle_text(msg)
            end
        end
        mon_print("Stream ended.")
        print("LiveCC: stream ended.")
    end,

    -- Audio player: drains the queue; may block on speaker_audio_empty
    function()
        if #speakers == 0 then return end
        while true do
            if #audio_queue > 0 then
                play_on_all(table.remove(audio_queue, 1))
            else
                os.pullEvent("livecc_audio")   -- sleep until a chunk arrives
            end
        end
    end,

    -- Key watcher
    function()
        while true do
            local _, key = os.pullEvent("key")
            if key == keys.q then
                if ws then ws.close() end
                mon_print("Stopped.")
                print("LiveCC: stopped by user.")
                return
            end
        end
    end
)
