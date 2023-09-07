#pragma once

// from https://github.com/pichenettes/eurorack/blob/master/clouds/dsp/mu_law.h

namespace mu_law {
  inline unsigned char Lin2MuLaw(int16_t pcm_val) {
      int16_t mask;
      int16_t seg;
      uint8_t uval;
      pcm_val = pcm_val >> 2;
      if (pcm_val < 0) {
          pcm_val = -pcm_val;
          mask = 0x7f;
      } else {
          mask = 0xff;
      }
      if (pcm_val > 8159) pcm_val = 8159;
      pcm_val += (0x84 >> 2);

      if (pcm_val <= 0x3f) seg = 0;
      else if (pcm_val <= 0x7f) seg = 1;
      else if (pcm_val <= 0xff) seg = 2;
      else if (pcm_val <= 0x1ff) seg = 3;
      else if (pcm_val <= 0x3ff) seg = 4;
      else if (pcm_val <= 0x7ff) seg = 5;
      else if (pcm_val <= 0xfff) seg = 6;
      else if (pcm_val <= 0x1fff) seg = 7;
      else seg = 8;
      if (seg >= 8)
          return static_cast<uint8_t>(0x7f ^ mask);
      else {
          uval = static_cast<uint8_t>((seg << 4) | ((pcm_val >> (seg + 1)) & 0x0f));
          return (uval ^ mask);
      }
  }

  short MuLaw2Lin(uint8_t u_val) {
      int16_t t;
      u_val = ~u_val;
      t = ((u_val & 0xf) << 3) + 0x84;
      t <<= ((unsigned)u_val & 0x70) >> 4;
      return ((u_val & 0x80) ? (0x84 - t) : (t - 0x84));
  }

  #define LUT_ULAW_SIZE 256
  int16_t lut_ulaw[LUT_ULAW_SIZE];

  void PopulateLUT() {
    for(int i = 0; i < LUT_ULAW_SIZE; i++) {
      lut_ulaw[i] = MuLaw2Lin(i);
    }
  }

  inline int32_t Clip16(int32_t x) {
    if (x < -32768) {
      return -32768;
    } else if(x > 32767) {
      return 32767;
    } else {
      return x;
    }
  }

  uint8_t Encode(float sample) {
    int16_t as_int = Clip16(static_cast<int32_t>(sample * 32768.0f));
    return Lin2MuLaw(as_int);
  }

  float Decode(uint8_t sample) {
    int16_t decoded = lut_ulaw[sample];
    const float scale = 1.0f / 32768.0f;
    return scale * decoded;
  }

}
