/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// We refer https://github.com/andravin/wincnn to access the winograd transform matrixs
#pragma once

#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {
namespace math {

static float weight_transform_matrix[8][3] = {
  {1.f, 0.f, 0.f}, {-2.f/9, -2.f/9, -2.f/9},
  {-2.f/9, 2.f/9, -2.f/9}, {1.f/90, 1.f/45, 2.f/45},
  {1.f/90, -1.f/45, 2.f/45}, {32.f/45, 16.f/45, 8.f/45},
  {32.f/45, -16.f/45, 8.f/45}, {0.f, 0.f, 1.f}
};

template<>
void winograd_transform_weight<8, 3>(const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // weight shape is [out_channel, in_channel, kernel_h, kernel_w]
  int out_channel = weight.dims()[0];
  int in_channel = weight.dims()[1];
  // reshape and alloc transformed weight
  framework::DDim transformed_shape = framework::make_ddim(
          std::vector<int>{out_channel, in_channel, 64});
  float *outptr = output->mutable_data<float>(transformed_shape);
  const float *inptr = weight.data<float>();
  const float (*trans)[3] = weight_transform_matrix;
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int ic = 0; ic < in_channel; ++ic) {
      size_t offset = oc * in_channel + ic;
      float *kout = outptr + offset * 64;
      const float *k = inptr + offset * 9;
      float gw[8][3];
      for (int i = 0; i < 8; ++i) {
        gw[i][0] = trans[i][0] * k[0] + trans[i][1] * k[3] + trans[i][2] * k[6];
        gw[i][1] = trans[i][0] * k[1] + trans[i][1] * k[4] + trans[i][2] * k[7];
        gw[i][2] = trans[i][0] * k[2] + trans[i][1] * k[5] + trans[i][2] * k[8];
      }
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j, ++kout) {
          *kout = gw[i][0] * trans[j][0] + gw[i][1] * trans[j][1] + gw[i][2] * trans[j][2];
        }
      }
    }
  }
}

template<>
void winograd_transform_input<8, 3>(const framework::Tensor &input,
                                    framework::Tensor *output) {
  // transform input to [c, roundup(h/6), roundup(w/6), 64] tiles
  int channel = input.dims()[1];
  int height = input.dims()[2];
  int width = input.dims()[3];
//  int h_tiles = (height + 5 - 2) / 6;
//  int w_tiles = (width + 5 - 2) / 6;
  int h_tiles = (height + 5) / 6;
  int w_tiles = (width + 5) / 6;
  framework::DDim transformed_shape = framework::make_ddim(
          std::vector<int>{channel, h_tiles, w_tiles, 64});
  float *outptr = output->mutable_data<float>(transformed_shape);
  memset(outptr, 0, channel * h_tiles * w_tiles * 64 * sizeof(float));
  const float *inptr = input.data<float>();
  DLOG << "channel: " << channel << ", "
       << "height: " << height << ", "
       << "width: " << width << ", "
       << "h_tiles: " << h_tiles << ", "
       << "w_tiles: " << w_tiles;
  // pack input to tiles
  for (int c = 0; c < channel; ++c) {
    int inter_h = (height - 2) / 6;
    int inter_w = (width - 2) / 6;
//    int inter_h = height / 6;
//    int inter_w = width / 6;
    int remain_h = height - (inter_h * 6);
    int remain_w = width - (inter_w * 6);
    DLOG << "inter_h = " << inter_h << ", inter_w = " << inter_w
         << ", remain_h = " << remain_h << ", remain_w = " << remain_w;
    const float *in0 = inptr + c * height * width;
    const float *in1 = in0 + width;
    const float *in2 = in1 + width;
    const float *in3 = in2 + width;
    const float *in4 = in3 + width;
    const float *in5 = in4 + width;
    const float *in6 = in5 + width;
    const float *in7 = in6 + width;
    float *out = outptr + c * h_tiles * w_tiles * 64;

    for (int h = 0; h < inter_h; ++h) {
      for (int w = 0; w < inter_w; ++w) {
        memcpy(out, in0, 8 * sizeof(float));
        memcpy(out + 8, in1, 8 * sizeof(float));
        memcpy(out + 16, in2, 8 * sizeof(float));
        memcpy(out + 24, in3, 8 * sizeof(float));
        memcpy(out + 32, in4, 8 * sizeof(float));
        memcpy(out + 40, in5, 8 * sizeof(float));
        memcpy(out + 48, in6, 8 * sizeof(float));
        memcpy(out + 56, in7, 8 * sizeof(float));
        in0 += 6;
        in1 += 6;
        in2 += 6;
        in3 += 6;
        in4 += 6;
        in5 += 6;
        in6 += 6;
        in7 += 6;
        out += 64;
      }
      // remain width
      if (remain_w > 0) {
        memcpy(out, in0,  remain_w * sizeof(float));
        memcpy(out + 8, in1,  remain_w * sizeof(float));
        memcpy(out + 16, in2,  remain_w * sizeof(float));
        memcpy(out + 24, in3,  remain_w * sizeof(float));
        memcpy(out + 32, in4,  remain_w * sizeof(float));
        memcpy(out + 40, in5,  remain_w * sizeof(float));
        memcpy(out + 48, in6,  remain_w * sizeof(float));
        memcpy(out + 56, in7,  remain_w * sizeof(float));
        in0 += remain_w;
        in1 += remain_w;
        in2 += remain_w;
        in3 += remain_w;
        in4 += remain_w;
        in5 += remain_w;
        in6 += remain_w;
        in7 += remain_w;
        out += 64;
      }
      in0 += 5 * width;
      in1 += 5 * width;
      in2 += 5 * width;
      in3 += 5 * width;
      in4 += 5 * width;
      in5 += 5 * width;
      in6 += 5 * width;
      in7 += 5 * width;
    }
    // remain height
    if (remain_h > 0) {
      for (int w = 0; w < inter_w; ++w) {
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out + rh * 8, in0 + rh * width, 8 * sizeof(float));
        }
        out += 64;
        in0 += 6;
      }
      // remain width
      if (remain_w > 0) {
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out + rh * 8, in0 + rh * width, remain_w * sizeof(float));
        }
      }
    }
  }
  // transform tiles, compute B_T * d(c, b) * B
  for (int c = 0; c < channel; ++c) {
    for (int tile = 0; tile < h_tiles * w_tiles; ++tile) {
      float *out = outptr + (c * h_tiles * w_tiles + tile) * 64;
      // compute B_T * d(c, b)
      float bd[8][8];
      for (int i = 0; i < 8; ++i) {
        // bd[0][i] = d[0][i] + (-21/4) * d[2][i] + (21/4) * d[4][i] + (-1) * d[6][i]
        bd[0][i] = out[i] - 5.25 * out[16 + i] + 5.25 * out[32 + i] - out[48 + i];
        // bd[1][i] = d[1][i] + d[2][i] + (-17/4) * d[3][i] + (-17/4) * d[4][i] + d[5][i] + d[6][i]
        bd[1][i] = out[8 + i] + out[16 + i] - 4.25 * out[24 + i] - 4.25 * out[32 + i] + out[40 + i] + out[48 + i];
        // bd[2][i] = (-1) * d[1][i] + d[2][i] + (17/4) * d[3][i] + (-17/4) * d[4][i] + (-1) * d[5][i] + d[6][i]
        bd[2][i] = -1.0 * out[8 + i] + out[16 + i] + 4.25 * out[24 + i] - 4.25 * out[32 + i] - out[40 + i] + out[48 + i];
        // bd[3][i] = (1/2) * d[1][i] + (1/4) * d[2][i] + (-5/2) * d[3][i] + (-5/4) * d[4][i] + 2 * d[5][i] + d[6][i]
        bd[3][i] = 0.5 * out[8 + i] + 0.25 * out[16 + i] + (-2.5) * out[24 + i] + (-1.25) * out[32 + i] + 2 * out[40 + i] + out[48 + i];
        // bd[4][i] = (-1/2) * d[1][i] + (1/4) * d[2][i] + (5/2) * d[3][i] + (-5/4) * d[4][i] + (-2) * d[5][i] + d[6][i]
        bd[4][i] = -0.5 * out[8 + i] + 0.25 * out[16 + i] + 2.5 * out[24 + i] - 1.25 * out[32 + i] - 2 * out[40 + i] + out[48 + i];
        // bd[5][i] = 2 * d[1][i] + 4 * d[2][i] + (-5/2) * d[3][i] + (-5) * d[4][i] + (1/2) * d[5][i] + d[6][i]
        bd[5][i] = 2 * out[8 + i] + 4 * out[16 + i] - 2.5 * out[24 + i] - 5 * out[32 + i] + 0.5 * out[40 + i] + out[48 + i];
        // bd[6][i] = -2 * d[1][i] + 4 * d[2][i] + (5/2) * d[3][i] + (-5) * d[4][i] + (-1/2) * d[5][i] + d[6][i]
        bd[6][i] = -2 * out[8 + i] + 4 * out[16 + i] + 2.5 * out[24 + i] - 5 * out[32 + i] - 0.5 * out[40 + i] + out[48 + i];
        // bd[7][i] = (-1) * d[1][i] + (21/4) * d[3][i] + (-21/4) * d[5][i] + d[7][i]
        bd[7][i] = -1.0 * out[8 + i] + 5.25 * out[24 + i] - 5.25 * out[40 + i] + out[56 + i];
      }
      // compute B_T * d(c, b) * B
      for (int i = 0; i < 8; ++i) {
        // out[i][0] = bd[i][0] + (-21/4) * bd[i][2] + (21/4) * bd[i][4] + (-1) * bd[i][6]
        out[i * 8 + 0] = bd[i][0] - 5.25 * bd[i][2] + 5.25 * bd[i][4] - bd[i][6];
        // out[i][0] = bd[i][1] + bd[i][2] + (-17/4) * bd[i][3] + (-17/4) * bd[i][4] + bd[i][5] + bd[i][6]
        out[i * 8 + 1] = bd[i][1] + bd[i][2] - 4.25 * bd[i][3] - 4.25 * bd[i][4] + bd[i][5] + bd[i][6];
        // out[i][2] = (-1) * bd[i][1] + bd[i][2] + (17/4) * bd[i][3] + (-17/4) * bd[i][4] + (-1) * bd[i][5] + bd[i][6]
        out[i * 8 + 2] = -1.0 * bd[i][1] + bd[i][2] + 4.25 * bd[i][3] - 4.25 * bd[i][4] - bd[i][5] + bd[i][6];
        // out[i][3] = (1/2) * d[i][1] + (1/4) * bd[i][2] + (-5/2) * bd[i][3] + (-5/4) * bd[i][4] + 2 * bd[i][5] + bd[i][6]
        out[i * 8 + 3] = 0.5 * bd[i][1] + 0.25 * bd[i][2] - 2.5 * bd[i][3] - 1.25 * bd[i][4] + 2 * bd[i][5] + bd[i][6];
        // out[i][4] = (-1/2) * bd[i][1] + (1/4) * bd[i][2] + (5/2) * bd[i][3] + (-5/4) * bd[i][4] + (-2) * bd[i][5] + bd[i][6]
        out[i * 8 + 4] = -0.5 * bd[i][1] + 0.25 * bd[i][2] + 2.5 * bd[i][3] - 1.25 * bd[i][4] - 2 * bd[i][5] + bd[i][6];
        // out[i][5] = 2 * bd[i][1] + 4 * bd[i][2] + (-5/2) * bd[i][3] + (-5) * bd[i][4] + (1/2) * bd[i][5] + bd[i][6]
        out[i * 8 + 5] = 2 * bd[i][1] + 4 * bd[i][2] - 2.5 * bd[i][3] - 5 * bd[i][4] + 0.5 * bd[i][5] + bd[i][6];
        // out[i][6] = -2 * bd[i][1] + 4 * bd[i][2] + (5/2) * bd[i][3] + (-5) * bd[i][4] + (-1/2) * bd[i][5] + bd[i][6]
        out[i * 8 + 6] = -2 * bd[i][1] + 4 * bd[i][2] + 2.5 * bd[i][3] - 5 * bd[i][4] - 0.5 * bd[i][5] + bd[i][6];
        // out[i][7] = (-1) * bd[i][1] + (21/4) * bd[i][3] + (-21/4) * bd[i][5] + bd[i][7]
        out[i * 8 + 7] = -1.0 * bd[i][1] + 5.25 * bd[i][3] - 5.25 * bd[i][5] + bd[i][7];
      }
    }
  }
}

template<>
void winograd_transform_output<8, 3>(const framework::Tensor &input,
                                     const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // input shape is [in_channel, h_tiles, w_tiles, 64]
  // weight shape is [out_channel, in_channel, 64]
  int in_channel = input.dims()[0];
  int h_tiles = input.dims()[1];
  int w_tiles = input.dims()[2];
  int tiles = h_tiles * w_tiles;
  int out_channel = weight.dims()[0];
  // compute U*V first
  framework::Tensor output_m;
  framework::DDim shape = framework::make_ddim(std::vector<int>{out_channel, tiles, 64});
  float *output_m_ptr = output_m.mutable_data<float>(shape);
  memset(output_m_ptr, 0, output_m.numel() * sizeof(float));
  const float *input_ptr = input.data<float>();
  const float *weight_ptr = weight.data<float>();
  for (int i = 0; i < out_channel; ++i) {
    for (int j = 0; j < tiles; ++j) {
      const float *w_ptr = weight_ptr + i * in_channel * 64;
      const float *in_ptr = input_ptr + j * 64;
      float *m_ptr = output_m_ptr + (i * tiles + j) * 64;
      for (int c = 0; c < in_channel; ++c) {
        for (int k = 0; k < 64; ++k) {
          m_ptr[k] += w_ptr[k] * in_ptr[k];
        }
        w_ptr += 64;
        in_ptr += tiles * 64;
      }
    }
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int tile = 0; tile < tiles; ++tile) {
      float *m = output_m_ptr + (oc * tiles + tile) * 64;
      // compute A_T * m
      float am[6][8];
      for (int i = 0; i < 8; ++i) {
        // am[0][i] = m[0][i] + m[1][i] + m[2][i] + m[3][i] + m[4][i] + m[5][i] + m[6][i]
        am[0][i] = m[i] + m[8 + i] +  m[16 + i] +  m[24 + i] +  m[32 + i] +  m[40 + i] +  m[48 + i];
        // am[1][i] = m[1][i] - m[2][i] + 2 * m[3][i] - 2 * m[4][i] + 0.5 * m[5][i] - 0.5 * m[6][i]
        am[1][i] = m[8 + i] - m[16 + i] + 2 * m[24 + i] - 2 * m[32 + i] + 0.5 * m[40 + i] - 0.5 * m[48 + i];
        // am[2][i] = m[1][i] + m[2][i] + 4 * m[3][i] + 4 * m[4][i] + 0.25 * m[5][i] + 0.25 * m[6][i]
        am[2][i] = m[8 + i] + m[16 + i] + 4 * m[24 + i] + 4 * m[32 + i] + 0.25 * m[40 + i] + 0.25 * m[48 + i];
        // am[3][i] = m[1][i] - m[2][i] + 8 * m[3][i] - 8 * m[4][i] + 0.125 * m[5][i] - 0.125 * m[6][i]
        am[3][i] = m[8 + i] - m[16 + i] + 8 * m[24 + i] - 8 * m[32 + i] + 0.125 * m[40 + i] - 0.125 * m[48 + i];
        // am[4][i] = m[1][i] + m[2][i] + 16 * m[3][i] + 16 * m[4][i] + 0.0625 * m[5][i] + 0.0625 * m[6][i]
        am[4][i] = m[8 + i] + m[16 + i] + 16 * m[24 + i] + 16 * m[32 + i] + 0.0625 * m[40 + i] + 0.0625 * m[48 + i];
        // am[5][i] = m[1][i] - m[2][i] + 32 * m[3][i] - 32 * m[4][i] + 0.03125 * m[5][i] - 0.03125 * m[6][i] + m[7][i]
        am[5][i] = m[8 + i] - m[16 + i] + 32 * m[24 + i] - 32 * m[32 + i] + 0.03125 * m[40 + i] - 0.03125 * m[48 + i] + m[56 + i];
      }
      // compute A_T * m * A
      for (int i = 0; i < 6; ++i) {
        m[i * 8] = am[i][0] + am[i][1] + am[i][2] + am[i][3] + am[i][4] + am[i][5] + am[i][6];
        m[i * 8 + 1] = am[i][1] - am[i][2] + 2 * am[i][3] - 2 * am[i][4] + 0.5 * am[i][5] - 0.5 * am[i][6];
        m[i * 8 + 2] = am[i][1] + am[i][2] + 4 * am[i][3] + 4 * am[i][4] + 0.25 * am[i][5] + 0.25 * am[i][6];
        m[i * 8 + 3] = am[i][1] - am[i][2] + 8 * am[i][3] - 8 * am[i][4] + 0.125 * am[i][5] - 0.125 * am[i][6];
        m[i * 8 + 4] = am[i][1] + am[i][2] + 16 * am[i][3] + 16 * am[i][4] + 0.0625 * am[i][5] + 0.0625 * am[i][6];
        m[i * 8 + 5] = am[i][1] - am[i][2] + 32 * am[i][3] - 32 * am[i][4] + 0.03125 * am[i][5] - 0.03125 * am[i][6] + am[i][7];
      }
    }
  }

  int out_h = output->dims()[2];
  int out_w = output->dims()[3];
  float *output_ptr = output->mutable_data<float>();
  // copy valid region to final output
  for (int oc = 0; oc < out_channel; ++oc) {
    int inter_h = out_h / 6;
    int inter_w = out_w / 6;
    int remain_h = out_h - inter_h * 6;
    int remain_w = out_w - inter_w * 6;
    DLOG << "out_h = " << out_h << ", out_w = " << out_w;
    DLOG << "remain_h = " << remain_h << ", remain_w = " << remain_w;
    DLOG << "h_tiles = " << h_tiles << ", w_tiles = " << w_tiles;

    float *out_ptr0 = output_ptr + oc * out_h * out_w;
    float *out_ptr1 = out_ptr0 + out_w;
    float *out_ptr2 = out_ptr1 + out_w;
    float *out_ptr3 = out_ptr2 + out_w;
    float *out_ptr4 = out_ptr3 + out_w;
    float *out_ptr5 = out_ptr4 + out_w;
    const float *m_ptr = output_m_ptr + oc * tiles * 64;
    for (int tile_h = 0; tile_h < inter_h; ++tile_h) {
      for (int tile_w = 0; tile_w < inter_w; ++tile_w) {
        const float *m = m_ptr + (tile_h * w_tiles + tile_w) * 64;
        memcpy(out_ptr0, m, 6 * sizeof(float));
        memcpy(out_ptr1, m + 8, 6 * sizeof(float));
        memcpy(out_ptr2, m + 16, 6 * sizeof(float));
        memcpy(out_ptr3, m + 24, 6 * sizeof(float));
        memcpy(out_ptr4, m + 32, 6 * sizeof(float));
        memcpy(out_ptr5, m + 40, 6 * sizeof(float));
        out_ptr0 += 6;
        out_ptr1 += 6;
        out_ptr2 += 6;
        out_ptr3 += 6;
        out_ptr4 += 6;
        out_ptr5 += 6;
      }
      // remain w
      if (remain_w > 0) {
        const float *m = m_ptr + (tile_h * w_tiles + inter_w) * 64;
        memcpy(out_ptr0, m, remain_w * sizeof(float));
        memcpy(out_ptr1, m + 8, remain_w * sizeof(float));
        memcpy(out_ptr2, m + 16, remain_w * sizeof(float));
        memcpy(out_ptr3, m + 24, remain_w * sizeof(float));
        memcpy(out_ptr4, m + 32, remain_w * sizeof(float));
        memcpy(out_ptr5, m + 40, remain_w * sizeof(float));
        out_ptr0 += remain_w;
        out_ptr1 += remain_w;
        out_ptr2 += remain_w;
        out_ptr3 += remain_w;
        out_ptr4 += remain_w;
        out_ptr5 += remain_w;
      }
      out_ptr0 += 5 * out_w;
      out_ptr1 += 5 * out_w;
      out_ptr2 += 5 * out_w;
      out_ptr3 += 5 * out_w;
      out_ptr4 += 5 * out_w;
      out_ptr5 += 5 * out_w;
    }
    // remain h
    if (remain_h > 0) {
      for (int tile_w = 0; tile_w < inter_w; ++tile_w) {
        const float *m = m_ptr + (inter_h * w_tiles + tile_w) * 64;
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out_ptr0 + rh * out_w, m + rh * 8, 6 * sizeof(float));
        }
        out_ptr0 += 6;
      }
      if (remain_w > 0) {
        const float *m = m_ptr + (inter_h * w_tiles + inter_w) * 64;
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out_ptr0 + rh * out_w, m + rh * 8, remain_w * sizeof(float));
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
