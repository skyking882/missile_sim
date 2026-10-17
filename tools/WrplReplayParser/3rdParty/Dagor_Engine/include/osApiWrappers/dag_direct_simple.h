

#pragma once

// Compare filenames taking into consideration '*' and '?' symbols
// Returns 0, if filenames are equal, 1 otherwise
bool dd_fname_equal(const char *f1, const char *fn2);

// remove leading and trailing spaces, remove quotes and extra slashes,
// remove extra up-dirs (e.g. some\dir\..\tex => some\tex)
void dd_simplify_fname_c(char *fn);

// returns pointer to extension inside filename or NULL if no extension
// first character in returned string will be dot (.)
const char *dd_get_fname_ext(const char *pathname);

// returns pointer to filena with extension inside pathname
// or NULL for empty pathname
const char *dd_get_fname(const char *pathname);

// returns pointer to buf containing location of filename
// size of buffer must be sufficiently large (260 at max for Win32)
char *dd_get_fname_location(char *buf, const char *pathname);

// returns pointer to inside buf containing location of filename, without path and extenstion
// size of buffer must be sufficiently large (260 at max for Win32)
const char *dd_get_fname_without_path_and_ext(char *buf, int buf_size, const char *path);

// append slash, if there is no slash
// NOTE: target buffer must have enough space for this slash
void dd_append_slash_c(char *pathname);

int dd_stricmp(const char *a, const char *b);

int dd_strnicmp(const char *a, const char *b, int n);

extern const unsigned char *dd_local_cmp_lwtab;
extern const unsigned char *dd_local_cmp_uptab;