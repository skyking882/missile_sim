// Copyright (C) Gaijin Games KFT.  All rights reserved.

#include <ioSys/dag_baseIo.h>

#include "dag_assert.h"
#include "generic/dag_span.h"

IBaseSave::IBaseSave() : blocks() {
}

IBaseSave::~IBaseSave() {
}

void IBaseSave::beginBlock() {
    int ofs = 0;
    write(&ofs, sizeof(int));

    int o = tell();

    int i = append_items(blocks, 1);
    blocks[i].ofs = o;
}

void IBaseSave::endBlock(unsigned block_flags) {
    G_ASSERTF(block_flags <= 0x3, "block_flags={:#x}", block_flags); // 2 bits max
    if (blocks.size() <= 0)
        EXCEPTION("block not begun {}", tell());

    Block &b = blocks.back();
    int o = tell();
    seekto(b.ofs - (int) sizeof(int));
    int l = o - b.ofs;
    G_ASSERTF(l >= 0 && !(l & 0xC0000000), "o={:#x} b.ofs={:#x} l={:#x}", o, b.ofs, l);
    l |= (block_flags << 30);
    write(&l, sizeof(int));
    seekto(o);

    blocks.pop_back();
}

int IBaseSave::getBlockLevel() { return (int) blocks.size(); }

IBaseLoad::IBaseLoad() : blocks() {
}

IBaseLoad::~IBaseLoad() {
}

int IBaseLoad::beginBlock(unsigned *out_block_flags) {
    int l = 0;
    read(&l, sizeof(int));
    if (out_block_flags)
        *out_block_flags = (l >> 30) & 0x3u;
    l &= 0x3FFFFFFF;

    int o = tell();

    int i = append_items(blocks, 1);

    blocks[i].ofs = o;
    blocks[i].len = l;

    return l;
}

void IBaseLoad::endBlock() {
    if (blocks.size() <= 0)
        EXCEPTION("EndBlock without beginBlock {}", tell());

    Block &b = blocks.back();
    seekto(b.ofs + b.len);

    blocks.pop_back();
}

int IBaseLoad::getBlockLength() {
    if (blocks.size() <= 0)
        EXCEPTION("block not begun {}", tell());

    Block &b = blocks.back();
    return b.len;
}

int IBaseLoad::getBlockRest() {
    if (blocks.size() <= 0)
        EXCEPTION("block not begun {}", tell());

    Block &b = blocks.back();
    return b.ofs + b.len - tell();
}

int IBaseLoad::getBlockLevel() { return (int) blocks.size(); }


#include "osApiWrappers/dag_direct_simple.h"


static unsigned char uptab[256], lwtab[256];

const unsigned char *dd_local_cmp_lwtab = lwtab;
const unsigned char *dd_local_cmp_uptab = uptab;


void dd_init_local_conv(void) {
    int i;

    uptab[0] = lwtab[0] = 0;
    for (i = 1; i < 128; ++i) {
        uptab[i] = (unsigned char) std::toupper(i);
        lwtab[i] = (unsigned char) std::tolower(i);
    }
    for (i = 128; i < 256; ++i)
        uptab[i] = lwtab[i] = i;
}

static struct DDLocalConvInitializer {
    DDLocalConvInitializer() { dd_init_local_conv(); }
} dd_initializer;

#include "osApiWrappers/dag_pathDelim.h"

#if _TARGET_PC_WIN | _TARGET_XBOX | _TARGET_C1 | _TARGET_C2
static constexpr int DAGOR_MAX_PATH = 260;
#else
static constexpr int DAGOR_MAX_PATH = 516;
#endif

int dd_stricmp(const char *a, const char *b) {
    int d;

    for (;; ++a, ++b) {
        d = (int) uptab[(unsigned char) *a] - (int) uptab[(unsigned char) *b];
        if (d)
            return d;
        if (!*a || !*b)
            break;
    }
    return d;
}

int dd_strnicmp(const char *a, const char *b, int n) {
    int d;

    for (d = 0; n > 0; ++a, ++b, --n) {
        d = (int) uptab[(unsigned char) *a] - (int) uptab[(unsigned char) *b];
        if (d)
            return d;
        if (!*a || !*b)
            break;
    }
    return d;
}

void dd_simplify_fname_c(char *s) {
    if (!s)
        return;
    int i, len = (int) strlen(s);

    // check for URL format to prevent removal of ://
    if (char *semi = (char *) memchr(s, ':', len > 8 ? 8 : len))
        if (len > (semi + 3 - s) && semi[1] == '/' && semi[2] == '/') {
            len -= int(semi + 3 - s);
            s = semi + 3;
        }
    if (*s == '%') // skip named mount prefix if any
    {
        if (char *p = strchr(s + 1, '/')) {
            len -= int(p + 1 - s);
            s = p + 1;
        } else
            return;
    }

    for (;;) {
        bool ok = false;
        // remove leading spaces
        for (i = 0; i < len; ++i)
            if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
                break;
        if (i > 0) {
            len -= i;
            memmove(s, s + i, len + 1);
            ok = true;
        }
        // remove trailing spaces
        for (i = len - 1; i >= 0; --i)
            if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
                break;
        if (i < len - 1) {
            s[len = i + 1] = 0;
            ok = true;
        }
        // remove quotes
        for (i = 0; i < len; ++i)
            if (s[i] == '"')
                break;
        if (i < len) {
            int qs = i;
            for (++i; i < len; ++i)
                if (s[i] == '"')
                    break;
            if (i < len) {
                memmove(s + qs, s + qs + 1, len - qs);
                --len;
                --i;
                memmove(s + i, s + i + 1, len - i);
                --len;
                ok = true;
            }
        }
        if (!ok)
            break;
    }

    // replace all back slashes with normal ones
    for (i = 1; i < len; ++i)
        if (s[i] == PATH_DELIM_BACK)
            s[i] = PATH_DELIM;

    // remove extra slashes
    for (i = 1; i < len; ++i)
        if (s[i] == PATH_DELIM)
            if (s[i - 1] == PATH_DELIM) {
                memmove(s + i, s + i + 1, len - i);
                --len;
                --i;
            }

    // remove extra cur-dirs
    while (len > 1)
        if (s[0] == '.' && s[1] == PATH_DELIM) {
            len -= 2;
            memmove(s, s + 2, len + 1);
        } else
            break;

    for (i = 1; i < len; ++i)
        if (s[i] == '.' && (s[i - 1] == PATH_DELIM || s[i - 1] == ':'))
            if (s[i + 1] == PATH_DELIM || s[i + 1] == 0) {
                int e = 1;
                if (s[i + 1] == PATH_DELIM)
                    ++e;
                len -= e;
                memmove(s + i, s + i + e, len + 1 - i);
                --i;
            }

    // remove extra up-dirs
    for (i = 2; i < len; ++i)
        if (s[i - 2] == PATH_DELIM && s[i - 1] == '.' && s[i] == '.')
            if (s[i + 1] == PATH_DELIM || s[i + 1] == 0) {
                int j;
                for (j = i - 3; j >= 0; --j)
                    if (s[j] == PATH_DELIM || s[j] == ':')
                        break;

                if (++j < i - 2) {
                    int e = i + 1 - j;

                    if (e != 5 || s[j] != '.' || s[j + 1] != '.') {
                        if (s[i + 1] == PATH_DELIM)
                            ++e;

                        len -= e;
                        memmove(s + j, s + j + e, len + 1 - j);
                        i = std::max(j, 1);
                    }
                }
            }
}

bool dd_fname_equal(const char *fn1, const char *fn2) {
    if (!fn1 && !fn2)
        return true;
    if (!fn1 || !fn2)
        return false;

    char s1[DAGOR_MAX_PATH], s2[DAGOR_MAX_PATH];
    strncpy(s1, fn1, DAGOR_MAX_PATH - 1);
    s1[DAGOR_MAX_PATH - 1] = '\0';
    strncpy(s2, fn2, DAGOR_MAX_PATH - 1);
    s2[DAGOR_MAX_PATH - 1] = '\0';

    dd_simplify_fname_c(s1);
    dd_simplify_fname_c(s2);

    return dd_stricmp(s1, s2) == 0;
}


void dd_append_slash_c(char *fn) {
    if (!fn)
        return;
    int l = (int) strlen(fn);
    if (l > 0)
        if (fn[l - 1] != PATH_DELIM_BACK && fn[l - 1] != PATH_DELIM) {
            fn[l] = PATH_DELIM;
            fn[l + 1] = 0;
        }
}

const char *dd_get_fname_without_path_and_ext(char *buf, int buf_size, const char *path) {
    if (!path || !path[0])
        return path;
    strncpy(buf, path, buf_size);
    dd_simplify_fname_c(buf);
    char *ext = (char *) dd_get_fname_ext(buf);
    if (ext)
        *ext = '\0';
    char *delim = strrchr(buf, PATH_DELIM);
    if (!delim)
        delim = strrchr(buf, PATH_DELIM_BACK);
    return delim ? delim + 1 : buf;
}

char *dd_get_fname_location(char *buf, const char *filename) {
    if (strncmp(filename, "b64://", 6) == 0) {
        *buf = '\0';
        return buf;
    }
    if (buf != filename)
        strcpy(buf, filename);
    dd_simplify_fname_c(buf);

    char *p = strrchr(buf, PATH_DELIM);
    if (!p)
        p = strchr(buf, ':');

    if (p)
        p++;
    else
        p = buf;

    *p = '\0';
    return buf;
}

const char *dd_get_fname_ext(const char *path) {
    if (!path || !path[0])
        return NULL;

    const char *p = strrchr(path, PATH_DELIM_BACK);
    if (!p)
        p = path;

    const char *p1 = strrchr(p, PATH_DELIM);
    if (p1)
        p = p1;

    return strrchr(p, '.');
}

const char *dd_get_fname(const char *path) {
    if (!path || !path[0])
        return path;

    const char *p = strrchr(path, PATH_DELIM_BACK);
    if (p)
        p++;
    else
        p = path;

    const char *p1 = strrchr(p, PATH_DELIM);
    if (p1)
        p = p1 + 1;

    return p;
}