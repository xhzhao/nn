local SpatialAveragePooling, parent = torch.class('nn.SpatialAveragePoolingMKLDNN', 'nn.Module')

function SpatialAveragePooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.padW = padW or 0
   self.padH = padH or 0
   self.ceil_mode = false
   self.count_include_pad = true
   self.divide = true


   self:setEngine(1)

end

function SpatialAveragePooling:ceil()
   self.ceil_mode = true
   return self
end

function SpatialAveragePooling:floor()
   self.ceil_mode = false
   return self
end

function SpatialAveragePooling:setCountIncludePad()
   self.count_include_pad = true
   return self
end

function SpatialAveragePooling:setCountExcludePad()
   self.count_include_pad = false
   return self
end

local function backwardCompatible(self)
   if self.ceil_mode == nil then
      self.ceil_mode = false
      self.count_include_pad = true
      self.padH = 0
      self.padW = 0
   end
end

function SpatialAveragePooling:updateOutput(input)

   if self.timerEnable then
	startTime = sys.clock()
   end
   if sys and sys.initOk == 0 then
      self.initStep = 0
      self.mkldnnInitOk = 0
   end
   if self.initStep == 0 then
   	self.initStep = 1
      self.dnnPrimitives = torch.LongTensor(16)
   else
	self.mkldnnInitOk = 1
   end
   backwardCompatible(self)
   if self.compare  then
	   input.THNN.SpatialAveragePooling_updateOutput(
	      input:cdata(),
	      self.output:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode,
	      self.count_include_pad
	   )
	   tmpOut = torch.Tensor(self.output:size())
	   input.THNN.SpatialAveragePooling_MKLDNN_updateOutput(
	      input:cdata(),
	      tmpOut:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode,
	      self.count_include_pad,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )
	   outSize = tonumber(tmpOut:cdata().size[0]*tmpOut:cdata().size[1]*tmpOut:cdata().size[2]*tmpOut:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), outSize,8)
   else

	   input.THNN.SpatialAveragePooling_MKLDNN_updateOutput(
	      input:cdata(),
	      self.output:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode,
	      self.count_include_pad,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )

   end
   -- for backward compatibility with saved models
   -- which are not supposed to have "divide" field
   if not self.divide then
     self.output:mul(self.kW*self.kH)
   end

   if self.timerEnable then
        print("mkldnn SpatialAveragePooling forward time = ,",self.timeForward," backward time =",self.timeBackward)
        sys.avgpoolingTime_forward = sys.avgpoolingTime_forward + self.timeForward 
        sys.avgpoolingTime_backward = sys.avgpoolingTime_backward + self.timeBackward
        self.timeForward = sys.clock() - startTime
        self.cnt = self.cnt + 1
   end
   return self.output
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)


   if self.gradInput then

      if self.timerEnable then
	   startTime = sys.clock()
      end
	   if self.compare  then
	      input.THNN.SpatialAveragePooling_updateGradInput(
		 input:cdata(),
		 gradOutput:cdata(),
		 self.gradInput:cdata(),
		 self.kW, self.kH,
		 self.dW, self.dH,
		 self.padW, self.padH,
		 self.ceil_mode,
		 self.count_include_pad
	      )

		outSize = tonumber(self.gradInput:cdata().size[0] *self.gradInput:cdata().size[1] *self.gradInput:cdata().size[2] *self.gradInput:cdata().size[3])
		tmpOut = torch.Tensor(outSize)
		input.THNN.SpatialAveragePooling_MKLDNN_updateGradInput(
		 input:cdata(),
		 gradOutput:cdata(),
		 tmpOut:cdata(),
		 self.kW, self.kH,
		 self.dW, self.dH,
		 self.padW, self.padH,
		 self.ceil_mode,
		 self.count_include_pad,
		 self.dnnPrimitives:cdata(),self.mkldnnInitOk
		   )
		 input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.gradInput:cdata(), outSize,9)


	   else


	      input.THNN.SpatialAveragePooling_MKLDNN_updateGradInput(
		 input:cdata(),
		 gradOutput:cdata(),
		 self.gradInput:cdata(),
		 self.kW, self.kH,
		 self.dW, self.dH,
		 self.padW, self.padH,
		 self.ceil_mode,
		 self.count_include_pad,
		 self.dnnPrimitives:cdata(),self.mkldnnInitOk
	      )

	   end

      -- for backward compatibility
      if not self.divide then
         self.gradInput:mul(self.kW*self.kH)
      end
      if self.timerEnable then
	   self.timeBackward = sys.clock() - startTime
      end

      return self.gradInput
   end
end

function SpatialAveragePooling:__tostring__()
   local s = string.format('%s(%d,%d,%d,%d', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'
   return s 
end
