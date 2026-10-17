// Copyright (C) Gaijin Games KFT.  All rights reserved.

#include "ioSys/blk_shared.h"

bool DataBlock::allowSimpleString = false;

bool DataBlock::strongTypeChecking = false;
bool DataBlock::singleBlockChecking = false;
bool DataBlock::allowVarTypeChange = false;
bool DataBlock::fatalOnMissingFile = true;
bool DataBlock::fatalOnLoadFailed = true;
bool DataBlock::fatalOnBadVarType = true;
bool DataBlock::fatalOnMissingVar = true;

bool DataBlock::parseIncludesAsParams = false; // def= false;
bool DataBlock::parseOverridesNotApply = false; // def= false;
bool DataBlock::parseOverridesIgnored = false; // def= false;
bool DataBlock::parseCommentsAsParams = false; // def= false;

static thread_local DataBlock::IErrorReporterPipe *tls_reporter = nullptr;

static void issue_error_unhappy_path(bool do_fatal, int line, const char *errText);

#define issue_error(robust_load, do_fatal, line, ...)                         \
  do                                                                          \
  {                                                                           \
    if (DAGOR_UNLIKELY(!robust_load || tls_reporter))                         \
      issue_error_unhappy_path(do_fatal, line, fmt::format(__VA_ARGS__).c_str()); \
  } while (0)

void DataBlock::issue_error_missing_param(const char *pname, int type) const {
    issue_error(shared->blkRobustOps(), fatalOnMissingVar, __LINE__,
                "BLK param missing: block='{}', param='{}' in file '{}' (req type: {})", getBlockName(), pname,
                resolveFilename(),
                dblk::resolve_type(type));
}

void DataBlock::issue_error_missing_file(const char *fname, const char *desc) const {
    issue_error(shared->blkRobustLoad(), fatalOnMissingFile, __LINE__, "{}: '{}'", desc, fname);
}

void DataBlock::issue_error_load_failed(const char *fname, const char *desc) const {
    if (desc)
        issue_error(false, !shared->blkRobustLoad() && fatalOnMissingFile, __LINE__, "{}, '{}'", desc,
                fname ? fname : "");
    else
        issue_error(shared->blkRobustLoad(), fatalOnMissingFile, __LINE__, "{}, '{}'", "BLK read error", fname);
}

void DataBlock::issue_error_load_failed_ver(const char *fname, unsigned req_ver, unsigned file_ver) const {
    issue_error(false, !shared->blkRobustLoad() && fatalOnMissingFile, __LINE__,
                "BLK wrong format: {}, expected {}, '{}'", file_ver,
                req_ver, fname);
}

void DataBlock::issue_error_parsing(const char *fname, int curLine, const char *msg, const char *curLineP) const {
    issue_error(shared->blkRobustLoad(), DataBlock::fatalOnLoadFailed, __LINE__, "BLK error '{}',{}: {}:\n\n{}\n",
                fname ? fname : "<unknown>", curLine, msg, (curLineP && *curLineP) ? curLineP : "unknown");
}

void DataBlock::issue_error_bad_type(const char *pname, int type_new, int type_prev, const char *fname) const {
    issue_error(shared->blkRobustLoad(), fatalOnBadVarType, __LINE__,
                "BLK param '{}' (type {}) already exists with type {} in file '{}', block '{}'", pname,
                dblk::resolve_type(type_new),
                dblk::resolve_type(type_prev), fname, getBlockName());
}

void DataBlock::issue_error_bad_type(int pnid, int type_new, int type_prev) const {
    issue_error_bad_type(getName(pnid), type_new, type_prev, resolveFilename());
}

void DataBlock::issue_error_bad_value(const char *pname, const char *value, int type, const char *fname,
                                      int line) const {
    issue_error(shared->blkRobustLoad(), fatalOnLoadFailed, __LINE__,
                "BLK invalid {} (type {}) value in line {} of '{}': '{}'", pname,
                dblk::resolve_type(type), line, fname, value);
}

void DataBlock::issue_error_bad_type_get(int bnid, int pnid, int type_get, int type_data) const {
    if (type_get == TYPE_INT && type_data == TYPE_INT64) {
        int64_t v = getInt64ByNameId(pnid, 0);
        issue_error(shared->blkRobustOps(), false, __LINE__,
                    "BLK getInt() for int64=0x%X%08X: block='{}', param='{}'\nin file '{}'",
                    unsigned(v >> 32), unsigned(v), getName(bnid), getName(pnid), resolveFilename());
        return;
    }

    issue_error(shared->blkRobustOps(), strongTypeChecking, __LINE__,
                "BLK param wrong type: block='{}', param='{}'\nQueried <{}({})> but type is <{}({})>\nin file '{}'",
                getName(bnid), getName(pnid),
                dblk::resolve_type(type_get), dblk::resolve_short_type(type_get), dblk::resolve_type(type_data),
                dblk::resolve_short_type(type_data), resolveFilename());
}

void DataBlock::issue_deprecated_type_change(int pnid, int type_new, int type_prev) const {
    if (allowVarTypeChange || (type_prev == TYPE_INT && type_new == TYPE_INT64) || (
            type_prev == TYPE_INT64 && type_new == TYPE_INT))
        issue_error(shared->blkRobustOps(), false, __LINE__,
                "BLK deprecated type change for param '{}' from {} to {} (even if allowVarTypeChange=true) in block '{}'\n in file '{}'",
                getName(pnid), dblk::resolve_type(type_prev), dblk::resolve_type(type_new), getBlockName(),
                resolveFilename());
    else
        issue_error_bad_type(getName(pnid), type_new, type_prev, resolveFilename());
}

void DataBlock::issue_warning_huge_string(const char *pname, const char *value, const char *fname, int line) const {
    if (!shared->blkRobustLoad())
        LOGE("BLK parsed string for param '{}' is really long ({} bytes) in line {} of '{}': '{}'", pname,
             strlen(value), line, fname,
             value);
    G_UNUSED(pname);
    G_UNUSED(value);
    G_UNUSED(fname);
    G_UNUSED(line); // for case of DAGOR_DBGLEVEL<1 without forced logs
}

static void issue_error_unhappy_path(bool do_fatal, int line, const char *errText) {
    if (tls_reporter)
        return tls_reporter->reportError(errText, do_fatal);
    if (do_fatal)
        EXCEPTION("{}", errText);
    else {
        LOGE("{}", errText);
    }
}

DataBlock::InstallReporterRAII::InstallReporterRAII(DataBlock::IErrorReporterPipe *rep) {
    prev = tls_reporter;
    if (rep)
        tls_reporter = rep;
}

DataBlock::InstallReporterRAII::~InstallReporterRAII() { tls_reporter = prev; }

const char *DataBlock::resolveFilename(bool file_only) const {
    const char *s = shared->getSrc();
#if DAGOR_DBGLEVEL > 0
    if (file_only && s && strncmp(s, "BLK\n", 4) == 0)
        return "unknown";
#endif
    G_UNUSED(file_only);
    return (s && *s) ? s : (this == &emptyBlock ? "empty" : "unknown");
}

bool DataBlock::isValid() const { return shared->blkValid(); }

