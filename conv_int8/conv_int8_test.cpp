

void conv5x5s1_s8(const int8_t* input, const int8_t* weight, int32_t* output,
                  const int input_c, const int input_h, const int input_w,
                  const int output_c, const int output_h, const int output_w) {
  int image_size = input_h * input_w;
  int out_image_size = output_h * output_w;
  memset(output, 0, output_c * out_image_size * sizeof(int32_t));
  for (int oc = 0; oc < output_c; ++oc) {
    for (int ic = 0; ic < input_c; ++ic) {
      const int8_t* kernel = weight + (oc * input_c + ic) * 25;
      int32_t* output0 = output;
      int32_t* output1 = output + output_w;
      // load kernel
      asm volatile(
        "vld1.8    {d0-d3}, [%0]  \n"
        : "=r"(kernel),
        : // no output
        : "memory"
      );
      int oh = 0;
      for (; oh < output_h - 1; oh += 2) {
        const int8_t* r0 = input + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;
        const int8_t* r3 = r2 + input_w;
        const int8_t* r4 = r3 + input_w;
        const int8_t* r5 = r4 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
            "start=:                             \n"
            "vld1.8     {d4-d5}, [%[r0]]         \n"  // r0
            "add        [%[r0]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d0[0]            \n"
            "vmulsl.s8  q6, d6, d0[1]            \n"
            "vmulsl.s8  q7, d7, d0[2]            \n"
            "vdup.s8    d16, d0[3]               \n"
            "vdup.s8    d17, d0[4]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q14, d10, d12            \n"
            "vaddw.s16  q14, q14, d14            \n"
            "vaddl.s16  q15, d11, d13            \n"
            "vaddw.s16  q15, q15, d15            \n"

            "vld1.8     {d4-d5}, [%[r1]]         \n"  // r1
            "add        [%[r1]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d0[5]            \n"
            "vmulsl.s8  q6, d6, d0[6]            \n"
            "vmulsl.s8  q7, d7, d0[7]            \n"
            "vdup.s8    d18, d1[0]               \n"
            "vdup.s8    d19, d1[1]               \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vmulsl.s8  q5, d4, d0[0]            \n"  // next row
            "vmulsl.s8  q6, d6, d0[1]            \n"
            "vmulsl.s8  q7, d7, d0[2]            \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q10, d10, d12            \n"
            "vaddw.s16  q10, q10, d14            \n"
            "vaddl.s16  q11, d11, d13            \n"
            "vaddw.s16  q11, q11, d15            \n"

            "vld1.8     {d4-d5}, [%[r2]]         \n"  // r2
            "add        [%[r2]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d1[2]            \n"
            "vmulsl.s8  q6, d6, d1[3]            \n"
            "vmulsl.s8  q7, d7, d1[4]            \n"
            "vdup.s8    d16, d1[5]               \n"
            "vdup.s8    d17, d1[6]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vmulsl.s8  q5, d4, d0[5]            \n"  // next row
            "vmulsl.s8  q6, d6, d0[6]            \n"
            "vmulsl.s8  q7, d7, d0[7]            \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q10, q10, q12            \n"
            "vadd.s32   q11, q11, q13            \n"

            "vld1.8     {d4-d5}, [%[r3]]         \n"  // r3
            "add        [%[r3]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d1[7]            \n"
            "vmulsl.s8  q6, d6, d2[0]            \n"
            "vmulsl.s8  q7, d7, d2[1]            \n"
            "vdup.s8    d18, d2[2]               \n"
            "vdup.s8    d19, d2[3]               \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vmulsl.s8  q5, d4, d1[2]            \n"  // next row
            "vmulsl.s8  q6, d6, d1[3]            \n"
            "vmulsl.s8  q7, d7, d1[4]            \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q10, d10, d12            \n"
            "vaddw.s16  q10, q10, d14            \n"
            "vaddl.s16  q11, d11, d13            \n"
            "vaddw.s16  q11, q11, d15            \n"
 
            "vld1.8     {d4-d5}, [%[r4]]         \n"  // r4
            "add        [%[r4]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d2[4]            \n"
            "vmulsl.s8  q6, d6, d2[5]            \n"
            "vmulsl.s8  q7, d7, d2[6]            \n"
            "vdup.s8    d16, d2[7]               \n"
            "vdup.s8    d17, d3[1]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vmulsl.s8  q5, d4, d1[7]            \n"  // next row
            "vmulsl.s8  q6, d6, d1[0]            \n"
            "vmulsl.s8  q7, d7, d2[1]            \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q10, q10, q12            \n"
            "vadd.s32   q11, q11, q13            \n"

            "vld1.32    {d24-d27}, [%[output0]]  \n"
            "vadd.s32   q12, q12, q14            \n"
            "vadd.s32   q13, q13, q15            \n"
            "vst1.32    {d24-d27}, [%[output0]]! \n"
           
            "vld1.8     {d4-d5}, [%[r5]]         \n"  // row 5
            "add        [%[r5]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d2[4]            \n"
            "vmulsl.s8  q6, d6, d2[5]            \n"
            "vmulsl.s8  q7, d7, d2[6]            \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q10, q10, q12            \n"
            "vadd.s32   q11, q11, q13            \n"

            "vld1.32    {d24-d27}, [%[output1]]  \n"
            "vadd.s32   q12, q12, q10            \n"
            "vadd.s32   q13, q13, q11            \n"
            "vst1.32    {d24-d27}, [%[output1]]! \n"

            "subs       [%[ow]], #1              \n"
            "bne        start=                   \n"
            : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
              [r4] "+r"(r4), [r5] "+r"(r5), [ow] "+r"(ow),
              [output0] "+r"(output0), [output1] "+r"(output1)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
              "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
          );
        }
        if (remain > 0) {
          asm volatile(
            "start=:                             \n"
            "vld1.8     d4, [%[r0]]              \n"
            "vld1.8     d5, [%[r1]]              \n"
            "vld1.8     d6, [%[r2]]              \n"
            "vld1.8     d7, [%[r3]]              \n"
            "vld1.8     d8, [%[r4]]              \n"
            "vld1.8     d9, [%[r5]]              \n"
            "add        [%[r0]], #1              \n"
            "add        [%[r1]], #1              \n"
            "add        [%[r2]], #1              \n"
            "add        [%[r3]], #1              \n"
            "add        [%[r4]], #1              \n"
            "add        [%[r5]], #1              \n"
            "vext.8     d10, d0, d1, #5          \n"
            "vext.8     d11, d1, d2, #2          \n"
            "vext.8     d12, d1, d2, #7          \n"
            "vext.8     d13, d2, d3, #4          \n"

            "vmull.s8   q7, d4, d0               \n"
            "vmull.s8   q8, d5, d10              \n"
            "vmull.s8   q9, d6, d11              \n"
            "vmlal.s8   q8, d7, d12              \n"
            "vmlal.s8   q9, d8, d13              \n"
            "vaddl.s16  q10, d14, d16            \n"
            "vaddw.s16  q10, q10, d18            \n"
            "vadd.s32   d4, d20, d21             \n"
            "vaddl.s16  q10, d15, d17            \n"
            "vaddw.s16  q10, q10, d19            \n"
            "vdup.s32   d14, d4[0]               \n"
            "vdup.s32   d15, d4[1]               \n"
            "vadd.s32   d15, d15, d14            \n"
            "vdup.s32   d14, d20[0]              \n"
            "vadd.s32   d15, d15, d14            \n"

            "ldr        r6, [%[output0]]         \n"
            "vdup.s32   d14, r6                  \n"
            "vadd.s32   d15, d15, d14            \n"
            "vst1.32    d15[0], [%[output0]]!    \n"

            "vmull.s8   q7, d5, d0               \n"
            "vmull.s8   q8, d6, d10              \n"
            "vmull.s8   q9, d7, d11              \n"
            "vmlal.s8   q8, d8, d12              \n"
            "vmlal.s8   q9, d9, d13              \n"
            "vaddl.s16  q10, d14, d16            \n"
            "vaddw.s16  q10, q10, d18            \n"
            "vadd.s32   d4, d20, d21             \n"
            "vaddl.s16  q10, d15, d17            \n"
            "vaddw.s16  q10, q10, d19            \n"
            "vdup.s32   d14, d4[0]               \n"
            "vdup.s32   d15, d4[1]               \n"
            "vadd.s32   d15, d15, d14            \n"
            "vdup.s32   d14, d20[0]              \n"
            "vadd.s32   d15, d15, d14            \n"

            "ldr        r6, [%[output1]]         \n"
            "vdup.s32   d14, r6                  \n"
            "vadd.s32   d15, d15, d14            \n"
            "vst1.32    d15[0], [%[output1]]!    \n"
           
            "subs       [%[remain]], #1          \n"
            "bne        start=                   \n"
            : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
              [r4] "+r"(r4), [r5] "+r"(r5), [remain] "+r"(remain),
              [output0] "+r"(output0), [output1] "+r"(output1)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
              "q7", "q8", "q9", "q10", "r6"
          );
        }
        output0 += output_w;
        output1 += output_w;
      }
      // remain output height
      for (; oh < output_h; ++oh) {
        const int8_t* r0 = input + ic * image_size + oh * input_w;
        const int8_t* r1 = r0 + input_w;
        const int8_t* r2 = r1 + input_w;
        const int8_t* r3 = r2 + input_w;
        const int8_t* r4 = r3 + input_w;

        int ow = output_w >> 3;
        int remain = output_w & 0x7;
        if (ow > 0) {
          asm volatile(
            "start=:                             \n"
            "vld1.8     {d4-d5}, [%[r0]]         \n"  // r0
            "add        [%[r0]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d0[0]            \n"
            "vmulsl.s8  q6, d6, d0[1]            \n"
            "vmulsl.s8  q7, d7, d0[2]            \n"
            "vdup.s8    d16, d0[3]               \n"
            "vdup.s8    d17, d0[4]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q14, d10, d12            \n"
            "vaddw.s16  q14, q14, d14            \n"
            "vaddl.s16  q15, d11, d13            \n"
            "vaddw.s16  q15, q15, d15            \n"

            "vld1.8     {d4-d5}, [%[r1]]         \n"  // r1
            "add        [%[r1]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d0[5]            \n"
            "vmulsl.s8  q6, d6, d0[6]            \n"
            "vmulsl.s8  q7, d7, d0[7]            \n"
            "vdup.s8    d18, d1[0]               \n"
            "vdup.s8    d19, d1[1]               \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vld1.8     {d4-d5}, [%[r2]]         \n"  // r2
            "add        [%[r2]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d1[2]            \n"
            "vmulsl.s8  q6, d6, d1[3]            \n"
            "vmulsl.s8  q7, d7, d1[4]            \n"
            "vdup.s8    d16, d1[5]               \n"
            "vdup.s8    d17, d1[6]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vld1.8     {d4-d5}, [%[r3]]         \n"  // r3
            "add        [%[r3]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d1[7]            \n"
            "vmulsl.s8  q6, d6, d2[0]            \n"
            "vmulsl.s8  q7, d7, d2[1]            \n"
            "vdup.s8    d18, d2[2]               \n"
            "vdup.s8    d19, d2[3]               \n"
            "vmlal.s8   q6, d8, d18              \n"
            "vmlal.s8   q7, d9, d19              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vld1.8     {d4-d5}, [%[r4]]         \n"  // r4
            "add        [%[r4]], #8              \n"
            "vext.8     d6, d4, d5, #1           \n"
            "vext.8     d7, d4, d5, #2           \n"
            "vext.8     d8, d4, d5, #3           \n"
            "vext.8     d9, d4, d5, #4           \n"
            "vmulsl.s8  q5, d4, d2[4]            \n"
            "vmulsl.s8  q6, d6, d2[5]            \n"
            "vmulsl.s8  q7, d7, d2[6]            \n"
            "vdup.s8    d16, d2[7]               \n"
            "vdup.s8    d17, d3[1]               \n"
            "vmlal.s8   q6, d8, d16              \n"
            "vmlal.s8   q7, d9, d17              \n"
            "vaddl.s16  q12, d10, d12            \n"
            "vaddw.s16  q12, q12, d14            \n"
            "vaddl.s16  q13, d11, d13            \n"
            "vaddw.s16  q13, q13, d15            \n"
            "vadd.s32   q14, q14, q12            \n"
            "vadd.s32   q15, q15, q13            \n"

            "vld1.32    {d24-d27}, [%[output0]]  \n"
            "vadd.s32   q12, q12, q14            \n"
            "vadd.s32   q13, q13, q15            \n"
            "vst1.32    {d24-d27}, [%[output0]]! \n"
           
            "subs       [%[ow]], #1              \n"
            "bne        start=                   \n"
            : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
              [r4] "+r"(r4), [ow] "+r"(ow), [output0] "+r"(output0)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
              "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
          );
        }
        if (remain > 0) {
          asm volatile(
            "start=:                             \n"
            "vld1.8     d4, [%[r0]]              \n"
            "vld1.8     d5, [%[r1]]              \n"
            "vld1.8     d6, [%[r2]]              \n"
            "vld1.8     d7, [%[r3]]              \n"
            "vld1.8     d8, [%[r4]]              \n"
            "add        [%[r0]], #1              \n"
            "add        [%[r1]], #1              \n"
            "add        [%[r2]], #1              \n"
            "add        [%[r3]], #1              \n"
            "add        [%[r4]], #1              \n"
            "vext.8     d10, d0, d1, #5          \n"
            "vext.8     d11, d1, d2, #2          \n"
            "vext.8     d12, d1, d2, #7          \n"
            "vext.8     d13, d2, d3, #4          \n"

            "vmull.s8   q7, d4, d0               \n"
            "vmull.s8   q8, d5, d10              \n"
            "vmull.s8   q9, d6, d11              \n"
            "vmlal.s8   q8, d7, d12              \n"
            "vmlal.s8   q9, d8, d13              \n"
            "vaddl.s16  q10, d14, d16            \n"
            "vaddw.s16  q10, q10, d18            \n"
            "vadd.s32   d4, d20, d21             \n"
            "vaddl.s16  q10, d15, d17            \n"
            "vaddw.s16  q10, q10, d19            \n"
            "vdup.s32   d14, d4[0]               \n"
            "vdup.s32   d15, d4[1]               \n"
            "vadd.s32   d15, d15, d14            \n"
            "vdup.s32   d14, d20[0]              \n"
            "vadd.s32   d15, d15, d14            \n"

            "ldr        r6, [%[output0]]         \n"
            "vdup.s32   d14, r6                  \n"
            "vadd.s32   d15, d15, d14            \n"
            "vst1.32    d15[0], [%[output0]]!    \n"

            "subs       [%[remain]], #1          \n"
            "bne        start=                   \n"
            : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
              [r4] "+r"(r4), [remain] "+r"(remain), [output0] "+r"(output0)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
              "q7", "q8", "q9", "q10", "r6"
          );
        } 
      }
    }
    output += out_image_size;
  }
}
