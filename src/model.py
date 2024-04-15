import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (1, 256, 256) to  (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # to (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # to (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # to (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # to (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten() # to (512*8*8)
        )

        self.linear_in  = nn.Sequential(nn.Linear(512 * 8 * 8, 1024), nn.ReLU())
        self.linear_out = nn.Sequential(nn.Linear(1024, 512 * 8 * 8), nn.ReLU())

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), #  (512, 8, 8), output shape (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # output shape (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # output shape (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # output shape (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # output shape (1, 256, 256)
            nn.Sigmoid() # output pixel values in range [0,1]
        )

    def forward(self, x, return_only_embedding=False):

        x = x.unsqueeze(1)

        # encoder
        x = self.encoder(x) # (b, 256, 256) to (b, 1024)

        # bottleneck in
        x = self.linear_in(x)

        # return only embedding at inference
        if(return_only_embedding):
          return x

        # bottleneck out
        x = self.linear_out(x)

        # decoder
        x = self.decoder(x) # (b, 1024) to (b, 256, 256)

        return x
