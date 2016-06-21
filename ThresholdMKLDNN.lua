local Threshold, parent = torch.class('nn.ThresholdMKLDNN','nn.Module')

function Threshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
   self.dnnPrimitives = torch.LongTensor(3)
   self.mkldnnInitOk = 0
   self.compare = sys.compare
   self.timerEnable = sys.timerEnable
   self.timeForward = 0
   self.timeBackward = 0
   self.cnt = 0

   self:validateParameters()
end

function Threshold:updateOutput(input)
   if self.timerEnable then
	sys.tic()
   end
   self:validateParameters()
   if self.compare  then
	   input.THNN.Threshold_updateOutput(
	      input:cdata(),
	      self.output:cdata(),
	      self.threshold,
	      self.val,
	      self.inplace
	   )
	   tmpOut = torch.Tensor(self.output:size())
	   input.THNN.Threshold_MKLDNN_updateOutput(
	      input:cdata(),
	      tmpOut:cdata(),
	      self.threshold,
	      self.val,
	      self.inplace,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )
	   outSize = tonumber(tmpOut:cdata().size[0]*tmpOut:cdata().size[1]*tmpOut:cdata().size[2]*tmpOut:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), outSize,6)
   else

   input.THNN.Threshold_MKLDNN_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.threshold,
      self.val,
      self.inplace,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
   )
   end
   self.mkldnnInitOk = 1
   if self.timerEnable then
	if self.cnt >= 10 then 
		print("mkldnn Threshold forward time = ,",self.timeForward/self.cnt," backward time =",self.timeBackward/self.cnt)
	end
	self.timeForward = self.timeForward + sys.toc()
	self.cnt = self.cnt + 1
   end
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   if self.timerEnable then
	sys.tic()
   end
   self:validateParameters()
   if self.compare then
	   input.THNN.Threshold_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.threshold,
	      self.inplace
	   )
	   tmpGradInput = torch.Tensor(self.gradInput:size())
	   input.THNN.Threshold_MKLDNN_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      tmpGradInput:cdata(),
	      self.threshold,
	      self.inplace,
	      self.dnnPrimitives:cdata()
	   )
	   --print("mkldnn Threshold backward compare")
	   outSize = tonumber(tmpGradInput:cdata().size[0]*tmpGradInput:cdata().size[1]*tmpGradInput:cdata().size[2]*tmpGradInput:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpGradInput:cdata(), self.gradInput:cdata(), outSize,7)
   else
	   input.THNN.Threshold_MKLDNN_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.threshold,
	      self.inplace,
	      self.dnnPrimitives:cdata()
	   )
   end
   if self.timerEnable then
	self.timeBackward = self.timeBackward + sys.toc()
   end
   return self.gradInput
end

function Threshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end
