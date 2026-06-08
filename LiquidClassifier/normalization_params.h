// Auto-generated normalization parameters
#ifndef NORMALIZATION_PARAMS_H
#define NORMALIZATION_PARAMS_H

#define N_FEATURES 22

// Feature order:
//   [0] temperatura_C
//   [1] condutividade_uS
//   [2] pH
//   [3] spec_F1_415nm
//   [4] spec_F2_445nm
//   [5] spec_F3_480nm
//   [6] spec_F4_515nm
//   [7] spec_F5_555nm
//   [8] spec_F6_590nm
//   [9] spec_F7_630nm
//   [10] spec_F8_680nm
//   [11] spec_Clear
//   [12] spec_NIR
//   [13] acustico_freq_Hz
//   [14] condutividade_25C
//   [15] ratio_azul_vermelho
//   [16] ratio_verde_vermelho
//   [17] ratio_nir_clear
//   [18] ratio_violeta_laranja
//   [19] spectral_mean
//   [20] spectral_std
//   [21] spectral_range

const float feature_mean[] = {
  10.417140f, 1098.389217f, 4.298862f, 0.465803f, 0.525413f, 0.597171f,
  0.671248f, 0.717188f, 0.717515f, 0.654241f, 0.565121f, 0.614601f,
  0.501120f, 1416.556224f, 1694.102134f, 0.891765f, 1.024040f, 12.141194f,
  11.702928f, 0.602942f, 0.131929f, 0.409665f,
};

const float feature_scale[] = {
  7.965228f, 912.507020f, 0.923986f, 0.273349f, 0.256287f, 0.231119f,
  0.211665f, 0.203639f, 0.182783f, 0.165530f, 0.207872f, 0.214187f,
  0.123068f, 41.444758f, 1424.959950f, 0.277138f, 0.309395f, 1541.210325f,
  701.384235f, 0.185309f, 0.041304f, 0.122225f,
};

#endif // NORMALIZATION_PARAMS_H