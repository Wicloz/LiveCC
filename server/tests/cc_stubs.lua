-- Minimal ComputerCraft API stubs for running lua/player.lua under plain
-- Lua 5.1 (via lupa) in the test suite.  Everything observable is recorded on
-- the global `stub` table so tests can configure the "world" before loading
-- the player and inspect what it did afterwards.
--
-- Only the surface player.lua actually touches is stubbed; anything else
-- erroring loudly is a feature (it means the player grew a dependency the
-- suite doesn't model yet).

stub = {
    speakers = {},        -- name -> { dev=, played={samples...}, stopped=n, accept=bool }
    speaker_order = {},   -- deterministic peripheral.getNames() order
    files = {},           -- path -> content for the fs stub
    now_ms = 1000000,     -- os.epoch("utc"): tests advance this by hand
    events = {},          -- os.queueEvent log
    timers = 0,
    console = {},         -- print/printError log
    ws_inbox = {},        -- messages ws.receive() will hand out (FIFO)
    ws_sent = {},         -- messages the player sent
    ws_closed = false,
    monitor = { w = 20, h = 10, present = true },
    blits = {},           -- { {x=,y=,text=,fg=,bg=}, ... } from the screen
    palettes = 0,         -- setPaletteColour call count
    palette_colors = {},  -- CC colour constant -> last {r,g,b} it was set to
    flushes = 0,          -- setVisible(true) count
}

function stub.add_speaker(name)
    local s = { played = {}, stopped = 0, accept = true }
    s.dev = {
        playAudio = function(samples)
            if not s.accept then return false end
            s.played[#s.played + 1] = samples
            return true
        end,
        stop = function() s.stopped = s.stopped + 1 end,
    }
    stub.speakers[name] = s
    stub.speaker_order[#stub.speaker_order + 1] = name
end

-- Drop playback records accumulated so far (e.g. the dedupe probe's silence
-- bursts during load) so tests only see what THEY caused.
function stub.reset_speakers()
    for _, s in pairs(stub.speakers) do
        s.played = {}
        s.stopped = 0
    end
end

-- (count, mean) of speaker `name`'s i-th played buffer — summarised Lua-side
-- so tests don't crawl 4800-entry tables across the Python boundary.
function stub.stats(name, i)
    local buf = stub.speakers[name].played[i]
    if not buf then return nil end
    local sum = 0
    for k = 1, #buf do sum = sum + buf[k] end
    return #buf, sum / #buf
end

function stub.play_count(name)
    return #stub.speakers[name].played
end

function stub.stop_count(name)
    return stub.speakers[name].stopped
end

-- ── CC globals ──────────────────────────────────────────────────────────────

local function noop() end

local monitor = {
    setTextScale = noop,
    setCursorBlink = noop,
    getSize = function() return stub.monitor.w, stub.monitor.h end,
}

peripheral = {
    find = function(kind)
        if kind == "monitor" and stub.monitor.present then return monitor end
        return nil
    end,
    getNames = function()
        local names = {}
        for i, n in ipairs(stub.speaker_order) do names[i] = n end
        return names
    end,
    getType = function(name)
        if stub.speakers[name] then return "speaker" end
        return nil
    end,
    wrap = function(name)
        return stub.speakers[name] and stub.speakers[name].dev or nil
    end,
}

term = {   -- fallback surface when no monitor is present
    getSize = function() return 51, 19 end,
    setCursorBlink = noop,
}

window = {
    create = function(_parent, _x, _y, _w, _h, _visible)
        local cx, cy = 1, 1
        return {
            setVisible = function(v)
                if v then stub.flushes = stub.flushes + 1 end
            end,
            setCursorPos = function(x, y) cx, cy = x, y end,
            blit = function(text, fg, bg)
                stub.blits[#stub.blits + 1] =
                    { x = cx, y = cy, text = text, fg = fg, bg = bg }
            end,
            setPaletteColour = function(colour, r, g, b)
                stub.palettes = stub.palettes + 1
                stub.palette_colors[colour] = { r, g, b }
            end,
            clear = noop,
            write = noop,
            setBackgroundColor = noop,
            setTextColor = noop,
        }
    end,
}

fs = {
    exists = function(path) return stub.files[path] ~= nil end,
    open = function(path, _mode)
        local content = stub.files[path]
        if content == nil then return nil end
        return { readAll = function() return content end, close = noop }
    end,
}

textutils = {
    unserialize = function(s)
        local f = loadstring("return " .. s)
        if not f then return nil end
        local ok, v = pcall(f)
        if ok then return v end
        return nil
    end,
}

os = {   -- replaces Lua's os entirely, like CC does
    epoch = function(_kind) return stub.now_ms end,
    queueEvent = function(...) stub.events[#stub.events + 1] = { ... } end,
    startTimer = function(_s)
        stub.timers = stub.timers + 1
        return stub.timers
    end,
    pullEvent = function()
        error("os.pullEvent must not run during tests (event loop is not driven)")
    end,
}

http = {
    websocket = function(url)
        stub.ws_url = url
        return {
            receive = function()
                local msg = table.remove(stub.ws_inbox, 1)
                if msg == nil then error("stub ws inbox is empty") end
                return msg, true
            end,
            send = function(msg, _binary) stub.ws_sent[#stub.ws_sent + 1] = msg end,
            close = function() stub.ws_closed = true end,
        }
    end,
}

parallel = {
    waitForAny = function() error("parallel must not run during tests") end,
}

colors = { black = 32768, white = 1 }
keys = { q = 16 }

function print(...)
    local parts = {}
    for i = 1, select("#", ...) do parts[i] = tostring(select(i, ...)) end
    stub.console[#stub.console + 1] = table.concat(parts, "\t")
end

printError = print

-- The player only requires cc.audio.dfpwm.  The fake decoder is deliberately
-- recognisable: every decoded sample is 17, and the count is 8 per input byte
-- — tests use both to prove dfpwm chunks were routed through a decode.
-- Each make_decoder() call gets a distinct id, and every decode call records
-- which id served it (stub.dfpwm_decoders / stub.dfpwm_decode_calls) — tests
-- use this to prove a chunk sliced across multiple feed_roles() calls still
-- shares ONE decoder instance (spec §4.6: fresh state per CHUNK, not per
-- slice), not a fresh one per slice.
stub.dfpwm_decoders = 0
stub.dfpwm_decode_calls = {}

function require(mod)
    if mod == "cc.audio.dfpwm" then
        return {
            make_decoder = function()
                stub.dfpwm_decoders = stub.dfpwm_decoders + 1
                local id = stub.dfpwm_decoders
                return function(data)
                    stub.dfpwm_decode_calls[#stub.dfpwm_decode_calls + 1] = id
                    local out = {}
                    for i = 1, #data * 8 do out[i] = 17 end
                    return out
                end
            end,
        }
    end
    error("player.lua required an unstubbed module: " .. tostring(mod))
end
