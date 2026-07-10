#include "engine/resync.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "engine/byteio.hpp"

namespace ccmfplayer {

namespace {

// Window size for the forward marker scan in Resync() and the initial tail
// window in FindTailChunks(). 64 KiB comfortably spans several small chunks
// while keeping per-seek reads cheap.
constexpr std::size_t kScanWindow = 64 * 1024;

// A hard cap on how many links a single ChainFrom will follow, so a
// pathological input (e.g. a file that is all marker bytes) can't spin
// unbounded. Far above any real chunk count in a tail window.
constexpr std::size_t kMaxChainLinks = 1u << 20;

}  // namespace

bool TryReadHeaderAt(std::istream& in, std::uint64_t offset, std::uint64_t fileSize,
                     ChunkEntry& out) {
    if (offset + kChunkHeaderSize > fileSize) {
        return false;
    }
    std::array<std::byte, kChunkHeaderSize> buf{};
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset));
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!in || static_cast<std::size_t>(in.gcount()) != buf.size()) {
        return false;
    }
    if (std::to_integer<std::uint8_t>(buf[0]) != kChunkMarker) {
        return false;
    }
    const std::span<const std::byte> view(buf);
    ChunkEntry entry;
    entry.offset = offset;
    entry.pts = ReadLittleEndian<6>(view.subspan(1, 6));
    entry.length = static_cast<std::uint32_t>(ReadLittleEndian<3>(view.subspan(7, 3)));
    entry.type = std::to_integer<std::uint8_t>(buf[10]);
    entry.compression = std::to_integer<std::uint8_t>(buf[11]);
    if (entry.End() > fileSize) {
        return false;  // declared payload overruns the file: not a real boundary
    }
    out = entry;
    return true;
}

ChainResult ChainFrom(std::istream& in, std::uint64_t start, std::uint64_t fileSize,
                      std::size_t maxLinks, std::vector<ChunkEntry>& out) {
    ChainResult result;
    std::uint64_t pos = start;
    while (result.links < maxLinks) {
        ChunkEntry entry;
        if (!TryReadHeaderAt(in, pos, fileSize, entry)) {
            break;
        }
        out.push_back(entry);
        ++result.links;
        const std::uint64_t next = entry.End();  // guaranteed <= fileSize by TryReadHeaderAt
        if (next == fileSize) {
            result.reachedEof = true;
            break;
        }
        pos = next;
    }
    return result;
}

std::optional<std::uint64_t> Resync(std::istream& in, std::uint64_t from, std::uint64_t limit,
                                    std::uint64_t fileSize, std::size_t requiredLinks,
                                    std::vector<ChunkEntry>& collected) {
    limit = std::min(limit, fileSize);
    std::vector<char> buf;
    std::uint64_t p = from;
    while (p < limit) {
        const std::uint64_t windowEnd = std::min<std::uint64_t>(p + kScanWindow, limit);
        const auto n = static_cast<std::size_t>(windowEnd - p);
        buf.resize(n);
        in.clear();
        in.seekg(static_cast<std::streamoff>(p));
        in.read(buf.data(), static_cast<std::streamsize>(n));
        const auto got = static_cast<std::size_t>(in.gcount());
        for (std::size_t i = 0; i < got; ++i) {
            if (static_cast<std::uint8_t>(buf[i]) != kChunkMarker) {
                continue;
            }
            const std::uint64_t candidate = p + i;
            std::vector<ChunkEntry> chain;
            const ChainResult r = ChainFrom(in, candidate, fileSize, requiredLinks, chain);
            if (r.reachedEof || r.links >= requiredLinks) {
                collected = std::move(chain);
                return candidate;
            }
        }
        if (got < n) {
            break;  // short read: nothing more to scan
        }
        p = windowEnd;
    }
    return std::nullopt;
}

std::optional<std::vector<ChunkEntry>> FindTailChunks(std::istream& in, std::uint64_t fileSize) {
    if (fileSize < kChunkHeaderSize) {
        return std::nullopt;
    }

    std::uint64_t window = kScanWindow;
    std::vector<char> buf;
    for (;;) {
        const bool wholeFile = window >= fileSize;
        const std::uint64_t start = wholeFile ? 0 : (fileSize - window);
        const auto n = static_cast<std::size_t>(fileSize - start);
        buf.resize(n);
        in.clear();
        in.seekg(static_cast<std::streamoff>(start));
        in.read(buf.data(), static_cast<std::streamsize>(n));
        const auto got = static_cast<std::size_t>(in.gcount());

        // Ascending scan: the earliest true boundary in the window chains
        // through every following real chunk to EOF, so it yields the longest
        // EOF-reaching chain. A false marker that happens to chain to EOF in
        // one hop is superseded; requiring >= 2 links (unless we've read the
        // whole file, where a lone final chunk is legitimate) rejects it.
        std::vector<ChunkEntry> best;
        for (std::size_t i = 0; i < got; ++i) {
            if (static_cast<std::uint8_t>(buf[i]) != kChunkMarker) {
                continue;
            }
            const std::uint64_t candidate = start + static_cast<std::uint64_t>(i);
            std::vector<ChunkEntry> chain;
            const ChainResult r = ChainFrom(in, candidate, fileSize, kMaxChainLinks, chain);
            if (r.reachedEof && chain.size() > best.size()) {
                best = std::move(chain);
                if (best.size() >= 2) {
                    break;  // earliest boundary found: longest possible chain
                }
            }
        }

        if (best.size() >= 2 || wholeFile) {
            if (best.empty()) {
                return std::nullopt;
            }
            return best;
        }
        window *= 4;  // last chunk larger than the window (or its predecessor is): widen
    }
}

}  // namespace ccmfplayer
