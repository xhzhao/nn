local Concat, parent = torch.class('nn.Concat', 'nn.Container')

function Concat:__init(dimension)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension = dimension

   self.timerEnable = sys.timerEnable
   self.timeForward = 0
   self.timeBackward1 = 0
   self.timeBackward2 = 0
   self.cnt = 0


end

function Concat:updateOutput(input)
   startTime = sys.clock()

   local outs = {}
   for i=1,#self.modules do
      local currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      outs[i] = currentOutput
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] + currentOutput:size(self.dimension)
      end
   end
   self.output:resize(self.size)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = outs[i]
      self.output:narrow(self.dimension, offset, currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end




if self.timerEnable then
                print("Concat forward time =         ,",self.timeForward," backward time =",self.timeBackward1+self.timeBackward2)
                sys.concatTime2 = sys.concatTime2 + (self.timeForward + self.timeBackward1+ self.timeBackward2)
        self.timeForward = (sys.clock() - startTime)
        self.cnt = self.cnt + 1
   end
   return self.output
end

function Concat:updateGradInput(input, gradOutput)

   startTime = sys.clock()
   self.gradInput:resizeAs(input)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)))

      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
      offset = offset + currentOutput:size(self.dimension)
   end

   if self.timerEnable then
        self.timeBackward1 =  ( sys.clock() - startTime)
   end


   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)

   startTime = sys.clock()
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      self:rethrowErrors(module, i, 'accGradParameters',
          input,
          gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
          scale)
      offset = offset + currentOutput:size(self.dimension)
   end
   if self.timerEnable then
        self.timeBackward2 =  sys.clock() - startTime
   end
end

function Concat:backward(input, gradOutput, scale)
   self.gradInput:resizeAs(input)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = self:rethrowErrors(module, i, 'backward', input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)), scale)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
      offset = offset + currentOutput:size(self.dimension)
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
