//
// Dagor Engine 6.5
// Copyright (C) Gaijin Games KFT.  All rights reserved.
//
#pragma once

#include <ioSys/dag_genIo.h>

struct ZSTD_CCtx_s;
struct ZSTD_DCtx_s;
struct ZSTD_CDict_s;
struct ZSTD_DDict_s;

class ZstdLoadFromMemCB : public IGenLoad {
    int fullReadInternal(void *ptr, int size);

public:
    ZstdLoadFromMemCB() = default;

    ZstdLoadFromMemCB(std::span<char> enc_data, const ZSTD_DDict_s *dict = nullptr, bool tmp = false) {
        initDecoder(enc_data, dict, tmp);
    }

    ~ZstdLoadFromMemCB() { termDecoder(); }

    bool initDecoder(std::span<char> enc_data, const ZSTD_DDict_s *dict, bool tmp = false);

    void termDecoder();

    void read(void *ptr, int size) override;

    int tryRead(void *ptr, int size) override;

    int tell() override {
        issueFatal();
        return 0;
    }

    void seekto(int) override { issueFatal(); }

    bool seekrel(int) override;

    int beginBlock(unsigned * = nullptr) override {
        issueFatal();
        return 0;
    }

    void endBlock() override { issueFatal(); }

    int getBlockLength() override {
        issueFatal();
        return 0;
    }

    int getBlockRest() override {
        issueFatal();
        return 0;
    }

    int getBlockLevel() override {
        issueFatal();
        return 0;
    }

    const char *getTargetName() override { return nullptr; }
    bool ceaseReading() override { return true; }

protected:
    ZSTD_DCtx_s *dstrm = nullptr;
    std::span<unsigned char> encDataBuf;
    unsigned encDataPos = 0;

    virtual bool supplyMoreData() { return false; }

    void issueFatal();

    inline int tryReadImpl(void *ptr, int size);
};

class ZstdLoadCB : public ZstdLoadFromMemCB {
public:
    ZstdLoadCB() = default; //-V730   /* since rdBuf shall not be filled in ctor for performance reasons */
    ZstdLoadCB(IGenLoad &in_crd, int in_size, const ZSTD_DDict_s *dict = nullptr, bool tmp = false) {
        open(in_crd, in_size, dict, tmp);
    }

    ~ZstdLoadCB() { close(); }

    const char *getTargetName() override { return loadCb ? loadCb->getTargetName() : NULL; }

    void open(IGenLoad &in_crd, int in_size, const ZSTD_DDict_s *dict = nullptr, bool tmp = false);

    void close();

    bool ceaseReading() override;

protected:
    static constexpr int RD_BUFFER_SIZE = (32 << 10);
    unsigned inBufLeft = 0;
    IGenLoad *loadCb = nullptr;
    alignas(16) char rdBuf[RD_BUFFER_SIZE];

    bool supplyMoreData() override;
};

class ZstdSaveCB : public IGenSave {
public:
    ZstdSaveCB(IGenSave &dest_cwr, int compression_level);

    ~ZstdSaveCB();

    void write(const void *ptr, int size) override;

    void finish();

    int tell() override {
        issueFatal();
        return 0;
    }

    void seekto(int) override { issueFatal(); }
    void seektoend(int) override { issueFatal(); }
    void beginBlock() override { issueFatal(); }
    void endBlock(unsigned) override { issueFatal(); }

    int getBlockLevel() override {
        issueFatal();
        return 0;
    }

    const char *getTargetName() override { return cwrDest ? cwrDest->getTargetName() : NULL; }

    void flush() override;

protected:
    static constexpr int BUFFER_SIZE = (32 << 10);

    IGenSave *cwrDest = nullptr;
    ZSTD_CCtx_s *zstdStream = nullptr;
    uint32_t wrBufUsed = 0, zstdBufUsed = 0, zstdBufSize = 0;
    uint8_t *wrBuf = nullptr;

    void issueFatal();

    void compress(const void *ptr, int sz);

    void compressBuffer() {
        compress(wrBuf, wrBufUsed);
        wrBufUsed = 0;
    }
};


class ZstdDecompressSaveCB : public IGenSave {
public:
    explicit ZstdDecompressSaveCB(IGenSave &dest_cwr, const ZSTD_DDict_s *dict = nullptr, bool tmp = false,
                                  bool multiframe = true);

    ~ZstdDecompressSaveCB();

    void write(const void *ptr, int size) override;

    void finish();

    int tell() override {
        issueFatal();
        return 0;
    }

    void seekto(int) override { issueFatal(); }
    void seektoend(int) override { issueFatal(); }
    void beginBlock() override { issueFatal(); }
    void endBlock(unsigned) override { issueFatal(); }

    int getBlockLevel() override {
        issueFatal();
        return 0;
    }

    const char *getTargetName() override { return cwrDest.getTargetName(); }

    void flush() override;

private:
    bool doProcessStep(void *opaqueInBuf);

protected:
    static constexpr int RD_BUFFER_SIZE = (16 << 10);
    IGenSave &cwrDest;
    ZSTD_DCtx_s *zstdStream = nullptr;
    int processedIn = 0;
    int processedOut = 0;
    bool isFrameFinished = true;
    bool isFinished = false;
    bool isBroken = false;
    bool allowMultiFrame = false;
    alignas(16) char rdBuf[RD_BUFFER_SIZE]; //-V730_NOINIT

    void issueFatal();
};


// will compress only sz data passed
// compressed size will be 2-3 bytes bigger, both compression and then decompression speed will be 5-10% faster (zstd 1.4.5)
// if max_window_log set > 0 then it is used as max window size (1<<max_window_log); 0 means using default for compression_level
int64_t zstd_stream_compress_data(IGenSave &dest, IGenLoad &src, const size_t sz, int compression_level,
                                  unsigned max_window_log = 0);

// if compression will read from src using tryRead until compression stream ends.compr_sz CAN NOT be zero, it is error
int64_t zstd_stream_decompress_data(IGenSave &dest, IGenLoad &src, const size_t compr_sz);

// if compression will read from src using tryRead untill compression stream ends
int64_t zstd_stream_decompress_data(IGenSave & dest, IGenLoad & src);

// will compress in one call, not using streaming. Difference in result is neglible (usually within 1-4 bytes, if any)
// minimize calls to tryRead/write, but allocates ~2*sz memory
// if max_window_log set > 0 then it is used as max window size (1<<max_window_log); 0 means using default for compression_level
int64_t zstd_compress_data_solid(IGenSave &dest, IGenLoad &src, const size_t sz, int compressionLevel = 18,
                                 unsigned max_window_log = 0);

// legacy API, but that's how it used to be, so otherwise we will get huge patch
inline int64_t zstd_compress_data(IGenSave &dest, IGenLoad &src, size_t sz, size_t solid_threshold = 1 << 20,
                                  int compressionLevel = 18, unsigned max_window_log = 0) {
    return (sz < solid_threshold)
               ? zstd_compress_data_solid(dest, src, sz, compressionLevel, max_window_log)
               : zstd_stream_compress_data(dest, src, sz, compressionLevel, max_window_log);
}

// old names
inline int64_t zstd_decompress_data(IGenSave &dest, IGenLoad &src, size_t compr_sz) {
    return zstd_stream_decompress_data(dest, src, compr_sz);
}

inline int64_t zstd_decompress_data(IGenSave &dest, IGenLoad &src) { return zstd_stream_decompress_data(dest, src); }

// Maximum compressed size in worst case single-pass scenario
size_t zstd_compress_bound(size_t srcSize);

size_t zstd_compress(void *dst, size_t maxDstSize, const void *src, size_t srcSize, int compressionLevel = 18,
                     unsigned max_window_log = 0);

size_t zstd_decompress(void *dst, size_t maxOriginalSize, const void *src, size_t compressedSize);

size_t zstd_compress_with_dict(ZSTD_CCtx_s *ctx, void *dst, size_t dstSize, const void *src, size_t srcSize,
                               const ZSTD_CDict_s *dict);

size_t zstd_decompress_with_dict(ZSTD_DCtx_s *ctx, void *dst, size_t dstSize, const void *src, size_t srcSize,
                                 const ZSTD_DDict_s *dict);

// trains dictionary buffer with sample_buf and sample_sizes and return size of used dictionary
size_t zstd_train_dict_buffer(std::span<char> dict_buf, int compressionLevel, std::span<char> sample_buf,
                              std::span<size_t> sample_sizes);

// creates dictionary from trained buffer (for compression), optionally reference dict_buf without copying
ZSTD_CDict_s *zstd_create_cdict(std::span<char> dict_buf, int compressionLevel, bool use_buf_ref = false);

// destroys compression dictionary
void zstd_destroy_cdict(ZSTD_CDict_s *dict);

// creates dictionary from trained buffer (for decompression), optionally reference dict_buf without copying
ZSTD_DDict_s *zstd_create_ddict(std::span<char> dict_buf, bool use_buf_ref = false);

// destroys decompression dictionary
void zstd_destroy_ddict(ZSTD_DDict_s *dict);

ZSTD_CCtx_s *zstd_create_cctx();

void zstd_destroy_cctx(ZSTD_CCtx_s *ctx);

ZSTD_DCtx_s *zstd_create_dctx(bool tmp = false);

void zstd_destroy_dctx(ZSTD_DCtx_s *ctx);

// compresses stream using dictionary (created with zstd_create_dict)
// if max_window_log set > 0 then it is used as max window size (1<<max_window_log); 0 means using default for compression_level
int64_t zstd_stream_compress_data_with_dict(IGenSave &dest, IGenLoad &src, const size_t sz, int cLev,
                                            const ZSTD_CDict_s *dict,
                                            unsigned max_window_log = 0);

// decompresses stream using dictionary (created with zstd_create_dict)
int64_t zstd_stream_decompress_data(IGenSave &dest, IGenLoad &src, const size_t compr_sz, const ZSTD_DDict_s *dict);
