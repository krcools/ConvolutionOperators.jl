# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `convolve!` for the banded `ConvOp` is now multithreaded and faster. Output rows
  are split into contiguous blocks, each handled by its own task. Within a block,
  memory is traversed contiguously and the banded inner loop is branch-free and
  `@simd`-friendly. Benchmarks (M = N, small P, 22 threads) show speedups from
  1.2x at M = 500 to 2.8x at M = 3000 compared with a row-threaded version of the
  previous loop. Start Julia with multiple threads (e.g. `julia -t auto`) to
  benefit. (#8)

### Fixed

- `convolve!` for `LeftRightMulCVO`: the space-time data `B * x` and `B * X` is
  converted back to a `Matrix` when `B` is a `LinearMap`, so downstream
  convolutions keep receiving matrix data. (#8)
- `eltype(::LeftRightMulCVO)` now promotes over the left linear map, the
  convolution operator and the right linear map. A duplicate definition that
  returned only the convolution operator's element type, and overrode the
  promoting one, was removed. (#8)
- Fixed `convolve!` for nested `LiftedConvOp`s. The input column indices are now
  composed into a single view (`axes(L,2)[colblock]`) instead of taking a view of
  a view. (#8)
- Test for `truncate` now checks the correct size of the truncated operator
  (`(2,2,4)`). (#8)
