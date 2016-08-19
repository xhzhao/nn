local SpatialConvolutionMM, parent = torch.class('nn.SpatialConvolutionMKLDNN', 'nn.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   --self.weight = torch.randn(nOutputPlane, nInputPlane*kH*kW) 
   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:setEngine(1)

   self:reset()
end

function SpatialConvolutionMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
	--self.weight = torch.ones(self.weight:size())
	--self.bias:zero()
	--self.weight = torch.ones(tonumber(self.weight:cdata().size[0]),tonumber(self.weight:cdata().size[1]))
	--self.bias = torch.ones(self.bias:cdata().size[0],self.bias:cdata().size[1])
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function SpatialConvolutionMM:updateOutput(input)


   if self.timerEnable then
	startTime = sys.clock()
   end
   if self.initStep == 0 then
   	self.initStep = 1
   else
	self.mkldnnInitOk = 1
   end
   if self.mkldnnInitOk == 0 then
      self.dnnPrimitives = torch.LongTensor(29)
   end

   self.finput = self.finput or input.new()
   self.fgradInput = self.fgradInput or input.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)

   if self.compare  then

	   input.THNN.SpatialConvolutionMM_updateOutput(
	      input:cdata(),
	      self.output:cdata(),
	      self.weight:cdata(),
	      self.bias:cdata(),
	      self.finput:cdata(),
	      self.fgradInput:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH
	   )
	   N =  input:cdata().size[0]
	   inC = input:cdata().size[1]
	   inH = input:cdata().size[2]
	   inW = input:cdata().size[3]

	   outC = self.weight:cdata().size[0]
	   outH = (inH + 2*self.padH - self.kH)/self.dH + 1
	   outW = (inW + 2*self.padW - self.kW)/self.dW + 1
	   
	   outSize = tonumber(N*outC*outH*outW)
   	   tmpOut = torch.Tensor(outSize)
	   input.THNN.SpatialConvolutionMM_MKLDNN_forward(
	      input:cdata(),
	      tmpOut:cdata(),
	      self.weight:cdata(),
	      self.bias:cdata(),
	      self.finput:cdata(),
	      self.fgradInput:cdata(),
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH
	   )

	   input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), tonumber(N*outC*outH*outW),1)
   else
	   input.THNN.SpatialConvolutionMM_MKLDNN_forward(
	      input:cdata(),
	      self.output:cdata(),
	      self.weight:cdata(),
	      self.bias:cdata(),
	      self.finput:cdata(),
	      self.fgradInput:cdata(),
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH
	   )
   end
   if self.timerEnable then
        print("mkldnn conv forward time =         ,",self.timeForward," backward time =",self.timeBackward1+self.timeBackward2)
        sys.convTime_forward = sys.convTime_forward + self.timeForward
        sys.convTime_backward = sys.convTime_backward + self.timeBackward1+ self.timeBackward2
        self.timeForward = (sys.clock() - startTime)
        self.cnt = self.cnt + 1
   end
   return self.output
end

function SpatialConvolutionMM:updateGradInput(input, gradOutput)
   --print "SpatialConvolutionMM:updateGradInput more log"

   if self.gradInput then

	   startTime = sys.clock()
	   input, gradOutput = makeContiguous(self, input, gradOutput)
	   if self.compare  then
	      			
			      input.THNN.SpatialConvolutionMM_updateGradInput(
				 input:cdata(),
				 gradOutput:cdata(),
				 self.gradInput:cdata(),
				 self.weight:cdata(),
				 self.finput:cdata(),
				 self.fgradInput:cdata(),
				 self.kW, self.kH,
				 self.dW, self.dH,
				 self.padW, self.padH
			      )
				--outSize = tonumber(input:cdata().size[0] *input:cdata().size[1] *input:cdata().size[2] *input:cdata().size[3])
				outSize = tonumber(self.gradInput:cdata().size[0] *self.gradInput:cdata().size[1] *self.gradInput:cdata().size[2] *self.gradInput:cdata().size[3])
		   		tmpOut = self.gradInput.new() --torch.Tensor(outSize)
			      input.THNN.SpatialConvolutionMM_MKLDNN_bwdData(
				 input:cdata(),
				 gradOutput:cdata(),
				 tmpOut:cdata(),
				 self.weight:cdata(),
				 self.bias:cdata(),
				 self.finput:cdata(),
				 self.fgradInput:cdata(),
				 self.dnnPrimitives:cdata(),self.mkldnnInitOk,
				 self.kW, self.kH,
				 self.dW, self.dH,
				 self.padW, self.padH
			      )

	      			input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.gradInput:cdata(), outSize,2)
	   else
			      input.THNN.SpatialConvolutionMM_MKLDNN_bwdData(
				 input:cdata(),
				 gradOutput:cdata(),
				 self.gradInput:cdata(),
				 self.weight:cdata(),
				 self.bias:cdata(),
				 self.finput:cdata(),
				 self.fgradInput:cdata(),
				 self.dnnPrimitives:cdata(),self.mkldnnInitOk,
				 self.kW, self.kH,
				 self.dW, self.dH,
				 self.padW, self.padH
			      )
	   end
   if self.timerEnable then
	self.timeBackward1 = (sys.clock() - startTime)
   end
   return self.gradInput
   end
end

function SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
   --print "SpatialConvolutionMM:accGradParameters"
   startTime = sys.clock()
   scale = scale or 1
   input, gradOutput = makeContiguous(self, input, gradOutput)
   if self.compare  then
		   input.THNN.SpatialConvolutionMM_accGradParameters(
		      input:cdata(),
		      gradOutput:cdata(),
		      self.gradWeight:cdata(),
		      self.gradBias:cdata(),
		      self.finput:cdata(),
		      self.fgradInput:cdata(),
		      self.kW, self.kH,
		      self.dW, self.dH,
		      self.padW, self.padH,
		      scale
		   )
   		outSize = tonumber(self.gradWeight:cdata().size[0] *self.gradWeight:cdata().size[1])
   		tmpOut = torch.Tensor(tonumber(self.gradWeight:cdata().size[0]),tonumber(self.gradWeight:cdata().size[1]))
		   input.THNN.SpatialConvolutionMM_MKLDNN_bwdFilter(
		      input:cdata(),
		      gradOutput:cdata(),
		      tmpOut:cdata(),
		      self.gradBias:cdata(),
		      self.finput:cdata(),
		      self.fgradInput:cdata(),
		      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
		      self.kW, self.kH,
		      self.dW, self.dH,
		      self.padW, self.padH,
		      scale
		   )
		input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.gradWeight:cdata(), outSize,3)

   else
		   input.THNN.SpatialConvolutionMM_MKLDNN_bwdFilter(
		      input:cdata(),
		      gradOutput:cdata(),
		      self.gradWeight:cdata(),
		      self.gradBias:cdata(),
		      self.finput:cdata(),
		      self.fgradInput:cdata(),
		      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
		      self.kW, self.kH,
		      self.dW, self.dH,
		      self.padW, self.padH,
		      scale
		   )
   end
   if self.timerEnable then
	self.timeBackward2 =  (sys.clock() - startTime)
   end
end

function SpatialConvolutionMM:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function SpatialConvolutionMM:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

function SpatialConvolutionMM:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end

