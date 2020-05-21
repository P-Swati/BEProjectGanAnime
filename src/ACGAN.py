import torch
import torch.nn as nn
import torch.nn.ConvTranspose2d as CT2d
import torch.nn.BatchNorm2d as BN2d
import torch.nn.Conv2d as C2d

class MyConGANGen(nn.Module):
    def __init__(self, latentVectorSize, classVectorSize):
	
        super(MyConGANGen, self).__init__()
        
        self.lD = latentVectorSize
        self.cD = classVectorSize

        self.PackedLayersofGen = nn.Sequential(
			#layer 1
                    CT2d(in_channels = self.lD + self.cD, #concatenate latent and class vec
                                       out_channels = 1024, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    BN2d(1024),
                    nn.ReLU(inplace = True),
			#layer 2
                    CT2d(in_channels = 1024,
                                       out_channels = 512,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    BN2d(512),
                    nn.ReLU(inplace = True),
			#layer 3
                    CT2d(in_channels = 512,
                                       out_channels = 256,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    BN2d(256),
                    nn.ReLU(inplace = True),
			#layer 4
                    CT2d(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    BN2d(128),
                    nn.ReLU(inplace = True),
			#layer 5
                    CT2d(in_channels = 128,
                                       out_channels = 3,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1),
                    nn.Tanh()
                    )
        return
    
    def forward(self, ipVec, classVec):
        finalVec = torch.cat((ipVec, classVec), dim = 1)  # Concatenate noise and class vector.
        finalVec = finalVec.unsqueeze(2).unsqueeze(3)  # 2 for dimesions, 3 for color channels
        return self.PackedLayersofGen(finalVec)

class MyConGANDisc(nn.Module):
    def __init__(self, countOfClasses):
        super(MyConGANDisc, self).__init__()

        self.countOfClasses = countOfClasses
        self.PackedLayersOfDisc = nn.Sequential(
                    C2d(in_channels = 3, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
		
                    C2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    BN2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
		
                    C2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    BN2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
		
                    C2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    BN2d(1024),
                    nn.LeakyReLU(0.2, inplace = True)
                    )  
	
	
        self.bin_classifier = nn.Sequential(
                    C2d(in_channels = 1024, 
                        out_channels = 1, 
                        kernel_size = 4,
                        stride = 1),
                    nn.Sigmoid()
                    ) 
	
	
        self.extraBotNeck = nn.Sequential(
                    C2d(in_channels = 1024, 
                        out_channels = 512, 
                        kernel_size = 4,
                        stride = 1),
                    BN2d(512),
                    nn.LeakyReLU(0.2)
                    )
	
        self.multilableClassificationLayer = nn.Sequential(
                    nn.Linear(512, self.countOfClasses),
                    nn.Sigmoid()
                    )
        return
    
    def forward(self, ipBatch):
	
        extrFeat = self.PackedLayersOfDisc(ipBatch)  
        realOrFake = self.bin_classifier(extrFeat).view(-1) 
        flatten = self.extraBotNeck(extrFeat).squeeze()
        multiLabOutput = self.multilableClassificationLayer(flatten)
        return realOrFake, multiLabOutput

if __name__ == '__main__':
    latentVectorSize = 100
    classVectorSize = 22
    batch = 5
    lVec = torch.randn(batch, latentVectorSize)
    cVec = torch.randn(batch, classVectorSize)
    
    GenObject = MyConGANGen(latentVectorSize, classVectorSize)
    DiscObject = MyConGANDisc(classVectorSize)
    o = GenObject(lVec, cVec)
#     print(o.shape)
    x, y = DiscObject(o)
#     print(x.shape, y.shape)
