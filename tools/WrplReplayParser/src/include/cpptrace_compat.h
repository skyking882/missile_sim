// src/include/cpptrace_compat.hpp
#pragma once
#if defined(__EMSCRIPTEN__)
#include <cstdint>
#include <string>
#include <vector>

namespace cpptrace { inline namespace v1 {
  using frame_ptr = std::uintptr_t;

  struct stacktrace {
    std::string to_string(bool = false) const { return {}; }
    void print() const {}
  };

  struct object_frame { frame_ptr raw_address{}; frame_ptr object_address{}; std::string object_path; };
  struct object_trace { std::vector<object_frame> frames; stacktrace resolve() const { return {}; } };
  struct safe_object_frame {
    frame_ptr raw_address{};
    frame_ptr address_relative_to_object_start{};
    char object_path[4097]{};
    object_frame resolve() const { return {}; }
  };

  inline stacktrace generate_trace(std::size_t = 0, std::size_t = 0) { return {}; }
  inline std::size_t safe_generate_raw_trace(frame_ptr*, std::size_t, std::size_t = 0, std::size_t = 0) { return 0; }
  inline void get_safe_object_frame(frame_ptr, safe_object_frame*) {}
  inline bool can_signal_safe_unwind() { return false; }
  inline bool can_get_safe_object_frame() { return false; }
}} // namespace cpptrace::v1
#else
#include <cpptrace/cpptrace.hpp>
#endif