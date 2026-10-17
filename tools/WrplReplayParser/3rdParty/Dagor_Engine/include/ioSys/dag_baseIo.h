//
// Dagor Engine 6.5
// Copyright (C) Gaijin Games KFT.  All rights reserved.
//
#pragma once

#include <ioSys/dag_genIo.h>

/// @addtogroup utility_classes
/// @{

/// @addtogroup serialization
/// @{


/// @file
/// Serialization callbacks.


/// Base implementation of interface class for writing to abstract output stream.
///
/// Implements some useful methods, but leaves a few virtual ones for
/// real implementation of output (write(), tellPos(), seekto(), seektoend()).
///
/// SaveException is thrown on write error, so there are no error return codes for methods.
class IBaseSave : public IGenSave {
public:
    IBaseSave();

    IBaseSave(const IBaseSave &) = default;

    IBaseSave(IBaseSave &&) = default;

    IBaseSave &operator=(const IBaseSave &) = default;

    IBaseSave &operator=(IBaseSave &&) = default;

    virtual ~IBaseSave();

    virtual void beginBlock();

    virtual void endBlock(unsigned block_flags_2bits = 0);

    virtual int getBlockLevel();

protected:
    /// @cond
    struct Block {
        int ofs;
    };

    std::vector<Block> blocks;
    /// @endcond
};


/// Base implementation of interface class for reading from abstract input stream.
///
/// Implements some useful methods, but leaves a few virtual ones for
/// real implementation of input.
///
/// LoadException is thrown on read error, so there are no error return codes for methods.
class IBaseLoad : public IGenLoad {
public:
    IBaseLoad();

    IBaseLoad(const IBaseLoad &) = default;

    IBaseLoad(IBaseLoad &&) = default;

    IBaseLoad &operator=(const IBaseLoad &) = default;

    IBaseLoad &operator=(IBaseLoad &&) = default;

    virtual ~IBaseLoad();

    virtual int beginBlock(unsigned *out_block_flags = nullptr);

    virtual void endBlock();

    virtual int getBlockLength();

    virtual int getBlockRest();

    virtual int getBlockLevel();

protected:
    /// @cond
    struct Block {
        int ofs, len;
    };

    std::vector<Block> blocks;
    /// @endcond
};

/// @}

/// @}
