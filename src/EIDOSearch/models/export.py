#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
import torch
from torch.onnx import OperatorExportTypes

from EIDOSearch.models import SegNetConv


def export_onnx(model, dummy, name):
    torch.save(model.state_dict(), f"{name}.pt")
    torch.onnx.export(model, dummy, f"{name}.onnx", opset_version=13, export_params=True,
                      keep_initializers_as_inputs=True, operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)


if __name__ == '__main__':
    model = SegNetConv(3, 1)
    export_onnx(model, torch.randn(1, 3, 224, 224), "segnet")
