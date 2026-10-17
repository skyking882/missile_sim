
#pragma once
#include "math/dag_TMatrix.h"
#include "reflection.h"
#include "ioSys/dag_dataBlock.h"
#include "ecs/entityId.h"
#include "mpi/types.h"
#include "mpi/codegen/serializers.h"

namespace danet {

  int TranslatedCoder(DANET_ENCODER_SIGNATURE);
  int WeirdFloatSerializer(DANET_ENCODER_SIGNATURE);
  int InvalidSerializer(DANET_ENCODER_SIGNATURE);
  int WeaponsCoder(DANET_ENCODER_SIGNATURE);
  int dummyVarForDamagedStateReflectionCoder(DANET_ENCODER_SIGNATURE);
  int UidCoder(DANET_ENCODER_SIGNATURE);
  template<>
  struct DefaultEncoderChooser<Invalid> {
    static constexpr reflection_var_encoder coder = InvalidSerializer;
  };
} // namespace danet
