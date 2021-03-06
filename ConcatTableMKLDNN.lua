local ConcatTable, parent = torch.class('nn.ConcatTableMKLDNN', 'nn.Container')

function ConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}

   self:setEngine(0)
  
end

function ConcatTable:updateOutput(input)
   local startTime
   if self.initStep == 0 then
   	self.initStep = 1
	self.dnnPrimitives = torch.LongTensor(20)
   else
	self.mkldnnInitOk = 1
   end

   for i=1,#self.modules do
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
   end
   if self.timerEnable then                                                                                        
        startTime = sys.clock()                                                                                    
   end

   for i=1,#self.modules do
      self:ConvertLayoutBackToNCHW(self.output[i],i)
   end

   if self.timerEnable then
      print("ConcatTable  forward time =         ",self.timeForward," backward time =",self.timeBackward)
      sys.concatTableTime_forward = sys.concatTableTime_forward + self.timeForward 
      sys.concatTableTime_backward = sys.concatTableTime_backward + self.timeBackward
      self.timeForward = sys.clock() - startTime
      self.cnt = self.cnt + 1
   end

   return self.output
end

local function retable(t1, t2, f)
   for k, v in ipairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = retable(t1[k] or {}, t2[k], f)
      else
         f(t1, k, v)
      end
   end
   for i=#t2+1, #t1 do
      t1[i] = nil
   end
   return t1
end

local function backward(self, method, input, gradOutput, scale)
   local isTable = torch.type(input) == 'table'
   local wasTable = torch.type(self.gradInput) == 'table'

   --[[
   if self.timerEnable then
        startTime = sys.clock()
   end]]
   local iterStartTime
   local iterBackward
   local backwardTime = 0
   if isTable then
      for i,module in ipairs(self.modules) do
         local currentGradInput = self:rethrowErrors(module, i, method, input, gradOutput[i], scale)
         if torch.type(currentGradInput) ~= 'table' then
            error"currentGradInput is not a table!"
         end
         if #input ~= #currentGradInput then
            error("table size mismatch: "..#input.." ~= "..#currentGradInput)
         end

         if self.timerEnable then
            iterStartTime = sys.clock()
         end
         if i == 1 then
            self.gradInput = wasTable and self.gradInput or {}
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  t[k] = t[k] or v:clone()
                  t[k]:resizeAs(v)
                  t[k]:copy(v)
               end
            )
         else
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  if t[k] then
                     t[k]:add(v)
                  else
                     t[k] = v:clone()
                  end
               end
            )
         end
         if self.timerEnable then
            iterBackward = sys.clock() - iterStartTime
            backwardTime = backwardTime + iterBackward
         end
      end
   else
      self.gradInput = (not wasTable) and self.gradInput or input:clone()
      for i,module in ipairs(self.modules) do
         local currentGradInput = self:rethrowErrors(module, i, method, input, gradOutput[i], scale)

         if self.timerEnable then
            iterStartTime = sys.clock()
         end

         if i == 1 then
            self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end

         if self.timerEnable then
            iterBackward = sys.clock() - iterStartTime
            backwardTime = backwardTime+ iterBackward
         end

      end
   end
   
   if self.timerEnable then
        self.timeBackward =  backwardTime
   end


   return self.gradInput
end

function ConcatTable:updateGradInput(input, gradOutput)
   return backward(self, 'updateGradInput', input, gradOutput)
end

function ConcatTable:backward(input, gradOutput, scale)
   return backward(self, 'backward', input, gradOutput, scale)
end

function ConcatTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      self:rethrowErrors(module, i, 'accGradParameters', input, gradOutput[i], scale)
   end
end

function ConcatTable:accUpdateGradParameters(input, gradOutput, lr)
   for i,module in ipairs(self.modules) do
      self:rethrowErrors(module, i, 'accUpdateGradParameters', input, gradOutput[i], lr)
   end
end

function ConcatTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
