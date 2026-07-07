#include "engine/controls_math.hpp"

#include <gtest/gtest.h>

#include "engine/chunk.hpp"

namespace ccmfplayer {
namespace {

// --------------------------------------------------------------------------
// SeekBarFractionForMouseX
// --------------------------------------------------------------------------

TEST(SeekBarFractionForMouseX, ClampsBeforeAndAfterTheBar) {
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(0.0, 100.0, 200.0), 0.0);
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(50.0, 100.0, 200.0), 0.0);
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(500.0, 100.0, 200.0), 1.0);
}

TEST(SeekBarFractionForMouseX, ExactEndpointsAndMidpoint) {
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(100.0, 100.0, 200.0), 0.0);
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(300.0, 100.0, 200.0), 1.0);
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(200.0, 100.0, 200.0), 0.5);
}

TEST(SeekBarFractionForMouseX, DegenerateBarWidthReturnsZero) {
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(150.0, 100.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(SeekBarFractionForMouseX(150.0, 100.0, -10.0), 0.0);
}

// --------------------------------------------------------------------------
// PtsForFraction / FractionForPts
// --------------------------------------------------------------------------

TEST(PtsForFraction, EndpointsAndMidpoint) {
    EXPECT_EQ(PtsForFraction(0.0, 100000), 0u);
    EXPECT_EQ(PtsForFraction(1.0, 100000), 100000u);
    EXPECT_EQ(PtsForFraction(0.5, 100000), 50000u);
}

TEST(PtsForFraction, ClampsOutOfRangeFractions) {
    EXPECT_EQ(PtsForFraction(-1.0, 100000), 0u);
    EXPECT_EQ(PtsForFraction(2.0, 100000), 100000u);
}

TEST(PtsForFraction, ZeroDurationAlwaysZero) {
    EXPECT_EQ(PtsForFraction(0.5, 0), 0u);
}

TEST(FractionForPts, EndpointsAndMidpoint) {
    EXPECT_DOUBLE_EQ(FractionForPts(0, 100000), 0.0);
    EXPECT_DOUBLE_EQ(FractionForPts(100000, 100000), 1.0);
    EXPECT_DOUBLE_EQ(FractionForPts(50000, 100000), 0.5);
}

TEST(FractionForPts, ClampsPastDuration) {
    EXPECT_DOUBLE_EQ(FractionForPts(999999, 100000), 1.0);
}

TEST(FractionForPts, ZeroDurationReturnsZeroNotNan) {
    EXPECT_DOUBLE_EQ(FractionForPts(12345, 0), 0.0);
}

TEST(PtsForFractionAndFractionForPts, RoundTripWithinRoundingTolerance) {
    constexpr std::uint64_t duration = 216000;
    for (double fraction : {0.0, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0}) {
        const std::uint64_t pts = PtsForFraction(fraction, duration);
        const double back = FractionForPts(pts, duration);
        EXPECT_NEAR(back, fraction, 1.0 / static_cast<double>(duration))
            << "fraction " << fraction;
    }
}

// --------------------------------------------------------------------------
// FormatTimecode
// --------------------------------------------------------------------------

TEST(FormatTimecode, ZeroIsZeroZero) {
    EXPECT_EQ(FormatTimecode(0), "0:00");
}

TEST(FormatTimecode, SecondsOnlyUnderAMinute) {
    EXPECT_EQ(FormatTimecode(5 * kSampleRate), "0:05");
    EXPECT_EQ(FormatTimecode(59 * kSampleRate), "0:59");
}

TEST(FormatTimecode, MinutesAndSecondsUnderAnHour) {
    EXPECT_EQ(FormatTimecode(65ull * kSampleRate), "1:05");
    EXPECT_EQ(FormatTimecode(3599ull * kSampleRate), "59:59");
}

TEST(FormatTimecode, HoursMinutesSecondsAllZeroPaddedExceptHours) {
    EXPECT_EQ(FormatTimecode(3600ull * kSampleRate), "1:00:00");
    EXPECT_EQ(FormatTimecode(3661ull * kSampleRate), "1:01:01");
    EXPECT_EQ(FormatTimecode((3600ull * 25 + 61) * kSampleRate), "25:01:01");
}

TEST(FormatTimecode, TruncatesPartialSeconds) {
    // 5.9 seconds worth of samples should still read "0:05", not round up.
    EXPECT_EQ(FormatTimecode(5 * kSampleRate + kSampleRate / 2), "0:05");
}

}  // namespace
}  // namespace ccmfplayer
