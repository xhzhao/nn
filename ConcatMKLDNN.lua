local Concat, parent = torch.class('nn.ConcatMKLDNN', 'nn.Container')

function Concat:__init(dimension)
   parent.__init(self)
   self.outputSize = torch.LongStorage()
   self.dimension = dimension


   self:setEngine(0)

end

function Concat:updateOutput(input)
   self:updateForLoadSnapshot()
   self.outputSize = self.outputSize or torch.LongStorage()
   if self.initStep == 0 then
   	self.initStep = 1
	self.dnnPrimitives = torch.LongTensor(20)
	self.outputArray = torch.LongTensor(10)
	self.gradOutputArray = torch.LongTensor(10)
   else
	self.mkldnnInitOk = 1
   end
   local iterStartTime
   local iterForward
   local forwardTime = 0
   local outs = {}
   local outputTable = {}
   for i=1,#self.modules do
      local currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      outs[i] = currentOutput
      outputTable = currentOutput:cdata()
      --print("module type =",self.modules[i].modules[1],"module engine = ",self.modules[i].modules[1]:getEngine()," module.kW = ",self.modules[i].modules[1].kW)
      if self.timerEnable then
        iterStartTime = sys.clock()
      end
      --setup the array for MKLDNN
      input.THNN.Concat_MKLDNN_setupLongTensor(self.outputArray:cdata(), currentOutput:cdata(), i)
      --self:ConvertLayoutBackToNCHW(currentOutput, i)
      if i == 1 then
         self.outputSize:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.outputSize[self.dimension] = self.outputSize[self.dimension] + currentOutput:size(self.dimension)
      end
      if self.timerEnable then
        iterForward = sys.clock() - iterStartTime
        forwardTime = forwardTime + iterForward
      end
   end
      if self.timerEnable then
        iterStartTime = sys.clock()
      end
   self.output:resize(self.outputSize)

   -- use MKLDNN to concat
   input.THNN.Concat_MKLDNN_updateOutput(self.outputArray:cdata(), self.output:cdata(), tonumber(#self.modules),self.dnnPrimitives:cdata(),self.mkldnnInitOk)

    if self.timerEnable then
        iterForward = sys.clock() - iterStartTime
        forwardTime = forwardTime + iterForward
        print("Concat forward time =         ,",self.timeForward," backward time =",self.timeBackward1+self.timeBackward2)
        sys.concatTime_forward = sys.concatTime_forward +self.timeForward 
        sys.concatTime_backward = sys.concatTime_backward + (self.timeBackward1+ self.timeBackward2)
        self.timeForward = forwardTime
        self.cnt = self.cnt + 1
   end

   return self.output
end

function Concat:updateGradInput(input, gradOutput)

   self.gradInput:resizeAs(input)
   local gradOutputBuffer = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor()
      gradOutputPart:resizeAs(module.output)
      gradOutputBuffer[i] = gradOutputPart
      input.THNN.Concat_MKLDNN_setupLongTensor(self.gradOutputArray:cdata(), gradOutputPart:cdata(), i)
   end

   --split gradOutput to  gradOutputArray
   input.THNN.Concat_MKLDNN_backward_split(self.gradOutputArray:cdata(), gradOutput:cdata(), tonumber(#self.modules),self.dnnPrimitives:cdata(),self.mkldnnInitOk)

   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputBuffer[i]
      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutputPart)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
            self.gradInput:cdata().mkldnnLayout = currentGradInput:cdata().mkldnnLayout
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)

   --startTime = sys.clock()
   local iterStartTime
   local iterBackward
   local backwardTime = 0

   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do

      if self.timerEnable then
        iterStartTime = sys.clock()
      end
      local currentOutput = module.output
      local gradOutputPart = gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension))
      if self.timerEnable then
            iterBackward = sys.clock() - iterStartTime
            backwardTime = backwardTime+ iterBackward
      end

      self:rethrowErrors(module, i, 'accGradParameters',
          input,
          gradOutputPart,
          scale)
      offset = offset + currentOutput:size(self.dimension)
   end
   if self.timerEnable then
        self.timeBackward2 =  backwardTime
   end
end

function Concat:backward(input, gradOutput, scale)
   self.gradInput:resizeAs(input)
   local gradOutputBuffer = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor()
      gradOutputPart:resizeAs(module.output)
      gradOutputBuffer[i] = gradOutputPart
      input.THNN.Concat_MKLDNN_setupLongTensor(self.gradOutputArray:cdata(), gradOutputPart:cdata(), i)
   end

   --split gradOutput to  gradOutputArray
   input.THNN.Concat_MKLDNN_backward_split(self.gradOutputArray:cdata(), gradOutput:cdata(), tonumber(#self.modules),self.dnnPrimitives:cdata(),self.mkldnnInitOk)

   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputBuffer[i]
      local currentGradInput = self:rethrowErrors(module, i, 'backward', input, gradOutputPart, scale)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
            self.gradInput:cdata().mkldnnLayout = currentGradInput:cdata().mkldnnLayout
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function Concat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      self:rethrowErrors(module, i, 'accUpdateGradParameters',
          input,
          gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
          lr)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
