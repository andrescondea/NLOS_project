import torch
import torch.nn as nn

class DepthMap(nn.Module):
    def __init__(self):
        super(DepthMap, self).__init__()
        # Bloque 1 (Entrada: B×1×256×32×32)
        self.conv3d_2 = nn.Conv3d(1, 48, kernel_size=3, stride=1, padding=1)
        self.relu      = nn.ReLU(inplace=True)
        self.conv3d_3 = nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1)
        # Pooling de la rama 1: colapsa toda la dimensión temporal (D=256)
        self.pool_branch1 = nn.AvgPool3d(kernel_size=(256, 1, 1),
                                         stride=(256,   1, 1))
        # Pooling de la rama 2: reduce espaciotemporal (D→64, H→16, W→16)
        self.pool_branch2 = nn.AvgPool3d(kernel_size=(4, 2, 2),
                                         stride=(4, 2, 2))

        # Bloque 2 para la rama 2: entrada B×48×64×16×16
        self.conv3d_5 = nn.Conv3d(48, 96, kernel_size=3, stride=1, padding=1)
        self.conv3d_6 = nn.Conv3d(96, 96, kernel_size=3, stride=1, padding=1)
        # Poolings posteriores en bloque 2:
        #   rama 2.1: colapsa D desde 64→16
        self.pool2_branch1 = nn.AvgPool3d(kernel_size=(64, 1, 1),
                                          stride=(64, 1, 1))
        #   rama 2.2: espaciotemporal D→16, H→8, W→8
        self.pool2_branch2 = nn.AvgPool3d(kernel_size=(4, 2, 2),
                                          stride=(4, 2, 2))

        # Bloque 3 (entrada a esta rama: B×96×16×8×8)
        self.conv3d_8  = nn.Conv3d(96,  192, kernel_size=3, stride=1, padding=1)
        self.conv3d_9  = nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1)
        # Pooling bloque 3: reduce D→4, H→4, W→4
        self.pool3     = nn.AvgPool3d(kernel_size=(4,2,2),
                                      stride=(4,2,2))

        # Bloque 4 (entrada a esta rama: B×192×4×4×4)
        self.conv3d_11 = nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv3d_12 = nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1)
        # Pooling bloque 4, rama 4.1: colapsa D→1, H→4, W→4
        self.pool4_branch1 = nn.AvgPool3d(kernel_size=(4, 1, 1),
                                          stride=(4, 1, 1))
        # Pooling bloque 4, rama 4.2: espaciotemporal D→2, H→2, W→2 (para conv2d posterior)
        self.pool4_branch2 = nn.AvgPool3d(kernel_size=(2, 2, 2),
                                          stride=(2, 2, 2))

        # Bloque 5 (entrada a esta rama: B×384×1×4×4 para branch1, 
        #              B×384×2×2×2 para branch2, luego ambas se fusionan)
        self.conv3d_14 = nn.Conv3d(384, 768, kernel_size=3, stride=1, padding=1)
        self.conv3d_15 = nn.Conv3d(768, 768, kernel_size=3, stride=1, padding=1)

        # ------------------------
        # DECODER 2D (upsample)
        # ------------------------

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Tras fusionar canales, seguimos con la red 2D (siguiendo canal counts del paper)
        self.conv2d_1  = nn.Conv2d(768, 192, kernel_size=3, stride=1, padding=1)
        self.conv2d_2  = nn.Conv2d(576, 384, kernel_size=3, stride=1, padding=1)
        self.conv2d_3  = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv2d_4  = nn.Conv2d(384,  96, kernel_size=3, stride=1, padding=1)
        self.conv2d_5  = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv2d_6  = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv2d_7  = nn.Conv2d(192,  48, kernel_size=3, stride=1, padding=1)
        self.conv2d_8  = nn.Conv2d( 96,  96, kernel_size=3, stride=1, padding=1)
        self.conv2d_9  = nn.Conv2d( 96,  96, kernel_size=3, stride=1, padding=1)
        self.conv2d_10 = nn.Conv2d(96,  24, kernel_size=3, stride=1, padding=1)
        self.conv2d_11 = nn.Conv2d(24,  48, kernel_size=3, stride=1, padding=1)
        self.conv2d_12 = nn.Conv2d(48,  48, kernel_size=3, stride=1, padding=1)
        self.conv2d_13 = nn.Conv2d(48,  12, kernel_size=3, stride=1, padding=1)
        self.conv2d_14 = nn.Conv2d(12,  24, kernel_size=3, stride=1, padding=1)
        self.conv2d_15 = nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=1)
        self.conv2d_16 = nn.Conv2d(24,   1, kernel_size=3, stride=1, padding=1)

        # ------------------------
        # MLP FINAL
        # ------------------------
        self.flatten = nn.Flatten()
        self.dense1  = nn.Linear(4096, 4096)
        self.dense2  = nn.Linear(4096, 4096)

    def forward(self, x):
        # x: (B,1,256,32,32)
        # --------------------------
        # BLOQUE 1
        # --------------------------
        h = self.relu(self.conv3d_2(x))   # → (B,48,256,32,32)
        h = self.relu(self.conv3d_3(h))   # → (B,48,256,32,32)

        # split ramas 1 y 2
        # ↳ rama1: collapse temporal (256→1)
        b1 = self.pool_branch1(h)         # → (B,48,  1, 32,32)
        b1 = b1.squeeze(2)                # → (B,48, 32,32)

        # ↳ rama2: reduce a (D→64,H→16,W→16)
        b2 = self.pool_branch2(h)         # → (B,48, 64,16,16)

        # --------------------------
        # BLOQUE 2 (en rama2)
        # --------------------------
        b2 = self.relu(self.conv3d_5(b2))  # → (B,96,  64,16,16)
        b2 = self.relu(self.conv3d_6(b2))  # → (B,96,  64,16,16)

        # split ramas 2.1 y 2.2
        # ↳ rama2.1: collapse temporal (64→16)
        b22 = self.pool2_branch1(b2)       # → (B,96, 16,16,16)
        b22 = b22.squeeze(2)               # → (B,96, 16,16)

        # ↳ rama2.2: reduce espaciotemporal (64→16,16→8,16→8)
        b21 = self.pool2_branch2(b2)       # → (B,96, 16, 8, 8)

        # --------------------------
        # BLOQUE 3 (en rama2.2)
        # --------------------------
        b21 = self.relu(self.conv3d_8(b21))  # → (B,192, 16, 8, 8)
        b21 = self.relu(self.conv3d_9(b21))  # → (B,192, 16, 8, 8)
        b3  = self.pool3(b21)                # → (B,192, 4, 4, 4)

        # --------------------------
        # BLOQUE 4 (en rama3)
        # --------------------------
        b3 = self.relu(self.conv3d_11(b3))   # → (B,384, 4, 4, 4)
        b3 = self.relu(self.conv3d_12(b3))   # → (B,384, 4, 4, 4)

        # split ramas 4.1 y 4.2
        # ↳ rama4.1: collapse temporal (4→1), espacio(4→4,4→4)
        b41 = self.pool4_branch1(b3)         # → (B,384, 1, 4, 4)
        b41 = b41.squeeze(2)                 # → (B,384, 4, 4)

        # ↳ rama4.2: reduce espaciotemporal (4→2,4→2,4→2)
        b42 = self.pool4_branch2(b3)         # → (B,384, 2, 2, 2)
        # aplanamos para entrar a conv3d_14/conv3d_15:
        # B×384×2×2×2 → B×384×2×2×2 (ya coincide) 
        # queremos encajar conv3d_14 (384→768) con “depth=2”:
        b42 = self.relu(self.conv3d_14(b42)) # → (B,768, 2, 2, 2)
        b42 = self.relu(self.conv3d_15(b42)) # → (B,768, 2, 2, 2)
        # colapsamos la última dimensión “2” (temporal) para pasar a 2D:
        b42 = b42.mean(dim=2)                # → (B,768, 2, 2)

        # --------------------------
        # DECODER 2D: primer “skip concat” con b41 
        # (ahora en (B,384,4,4))
        # --------------------------
        d = self.upsample(b42)               # → (B,768, 4, 4)
        d = self.relu(self.conv2d_1(d))      # → (B,192, 4, 4)
        # concatenamos “b41” (B,384,4,4) → da (B,576,4,4)
        d = torch.cat([d, b41], dim=1)       # → (B,576, 4, 4)
        d = self.relu(self.conv2d_2(d))      # → (B,384, 4, 4)
        d = self.relu(self.conv2d_3(d))      # → (B,384, 4, 4)

        # upsample a 8×8
        d = self.upsample(d)                 # → (B,384, 8, 8)
        d = self.relu(self.conv2d_4(d))      # → (B, 96, 8, 8)
        # concatenamos “b22” (B,96,16,16) pero hay que reescalar:
        #   b22 está en 16×16, queremos 8×8. Así que primero reducimos b22 a 8×8:
        b22_down = nn.functional.interpolate(b22, size=(8,8), mode='bilinear', align_corners=False)
        d = torch.cat([d, b22_down], dim=1)  # → (B,96+96=192,8,8)
        d = self.relu(self.conv2d_5(d))      # → (B,192, 8, 8)
        d = self.relu(self.conv2d_6(d))      # → (B,192, 8, 8)

        # upsample a 16×16
        d = self.upsample(d)                 # → (B,192, 16,16)
        d = self.relu(self.conv2d_7(d))      # → (B, 48,16,16)
        # concatenamos “b1” (B,48,32,32) tras reducirlo a 16×16
        b1_down = nn.functional.interpolate(b1, size=(16,16), mode='bilinear', align_corners=False)
        d = torch.cat([d, b1_down], dim=1)   # → (B,48+48=96,16,16)
        d = self.relu(self.conv2d_8(d))      # → (B, 96,16,16)
        d = self.relu(self.conv2d_9(d))      # → (B, 96,16,16)

        # upsample a 32×32
        d = self.upsample(d)                 # → (B,96, 32,32)
        d = self.relu(self.conv2d_10(d))     # → (B,24, 32,32)

        # --------------------------
        # CONVS FINALES
        # --------------------------
        d = self.relu(self.conv2d_11(d))     # → (B,48, 32,32)
        d = self.relu(self.conv2d_12(d))     # → (B,48, 32,32)
        # upsample final a 64×64
        d = self.upsample(d)                 # → (B,48, 64,64)
        d = self.relu(self.conv2d_13(d))     # → (B,12, 64,64)
        d = self.relu(self.conv2d_14(d))     # → (B,24, 64,64)
        d = self.relu(self.conv2d_15(d))     # → (B,24, 64,64)
        d = self.conv2d_16(d)                # → (B, 1, 64,64)

        # --------------------------
        # MLP HEAD
        # --------------------------
        out = self.flatten(d)                # → (B, 4096)
        out = self.relu(self.dense1(out))    # → (B, 4096)
        out = self.dense2(out)               # → (B, 4096)
        out = out.view(-1, 64, 64)           # → (B,  64, 64)

        return out
