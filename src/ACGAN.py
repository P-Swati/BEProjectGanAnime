import torch
import torch.nn as nn
import torch.nn.ConvTranspose2d as CT2d
import torch.nn.BatchNorm2d as BN2d
import torch.nn.Conv2d as C2d

class MyConGANGen(nn.Module):
    def __init__(self, latentVectorSize, classVectorSize):
	
        super(MyConGANGen, self).__init__()
	
        self.weight=1
	self.bias=0
        self.lD = latentVectorSize*self.weight + self.bias
        self.cD = classVectorSize*self.weight + self.bias
	
	def concatenate(a,b):
		return a+b
	
        self.PackedLayersofGen = nn.Sequential(
			#layer 1
                    CT2d(in_channels = concatenate(self.lD,self.cD), #concatenate latent and class vec
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
	print("inside forward function")
        finalVec = torch.cat((ipVec, classVec), dim = 1)  # Concatenate noise and class vector.
	print("final appended vector to be fed =")
	print(finalVec)
        finalVec = finalVec.unsqueeze(2).unsqueeze(3)  # 2 for dimesions, 3 for color channels
	print("resized final vector =")
	print(finalVec)
	print("pass this to gen input layer and return obtained output")
        temp=self.PackedLayersofGen(finalVec)
	print(temp)
	return temp

class MyConDisc(nn.Module):
    def __init__(self, countOfClasses):
        super(MyConDisc, self).__init__()
	
	print("total number of classes:")
	print(countOfClasses)
	
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
	print("packing discriminator dense layers finished..")
	
        self.bin_classifier = nn.Sequential(
                    C2d(in_channels = 1024, 
                        out_channels = 1, 
                        kernel_size = 4,
                        stride = 1),
                    nn.Sigmoid()
                    ) 
	
	print("Binary classifier layer initialized..")
	
	
        self.extraBotNeck = nn.Sequential(
                    C2d(in_channels = 1024, 
                        out_channels = 512, 
                        kernel_size = 4,
                        stride = 1),
                    BN2d(512),
                    nn.LeakyReLU(0.2)
                    )
	print("added an extra layer to compensate for size of input to ouput channel")
	
        self.multilableClassificationLayer = nn.Sequential(
                    nn.Linear(512, self.countOfClasses),
                    nn.Sigmoid()
                    )
	print("multilable classifier layer created..")
        return
    
    def forward(self, ipBatch):
        extrFeat = self.PackedLayersOfDisc(ipBatch) 
	print("feature extracted")
	print(extrFeat)
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
    print("random vector:")
    print(lVec)
    print("class vector")
    print(cVec)
    GenObject = MyConGANGen(latentVectorSize, classVectorSize)
    DiscObject = MyConDisc(classVectorSize)
    o = GenObject(lVec, cVec)
#     print(o.shape)
    x, y = DiscObject(o)
#     print(x.shape, y.shape)
