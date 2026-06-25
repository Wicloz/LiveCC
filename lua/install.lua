-- LiveCC installer  (server URL is injected by the server before delivery)
-- Install on a CC computer:  wget run <server>/install

local SERVER = "{{SERVER}}"

print("Installing LiveCC from " .. SERVER .. " ...")

local resp, err = http.get(SERVER .. "/player")
if not resp then
    printError("Download failed: " .. (err or "unknown error"))
    return
end
local code = resp.readAll()
resp.close()

if not code or #code < 10 then
    printError("Download failed: empty response (is the server running?)")
    return
end

local f = fs.open("livecc", "w")
f.write(code)
f.close()

print("LiveCC installed as 'livecc'.")
print("Run 'livecc --help' to get started.")
