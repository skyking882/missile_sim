
#include <stdio.h>
#include <stdarg.h>
#include "enet/enet.h"

inline void enet_logf(const char *format, ...) {

  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);

  //std::fprintf(stdout, "\n");
}