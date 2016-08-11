local ConcatTable, parent = torch.class('nn.ConcatTable', 'nn.Container')

function ConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}

   self.timerEnable = sys.timerEnable
   self.timeForward = 0
   self.timeBackward = 0
   self.cnt = 0

  
end

function ConcatTable:updateOutput(input)
   --[[
   local conStartTime
   local startTime
   if self.timerEnable then
        startTime = sys.clock()
        conStartTime = startTime
   end]]


   for i=1,#self.modules do
      if self.timerEnable then
        iterStartTime = sys.clock()
      end
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      if self.timerEnable then
        iterForward = sys.clock() - iterStartTime
        print("concatable forward time of each module(iteration)", i, "=    ", iterForward)
      end
   end
   --[[]]
   if self.timerEnable then
      --local forwardTime = sys.clock() - conStartTime
      --print("Current concatable forward time =     ", forwardTime) 

      print("ConcatTable  forward time =         ",self.timeForward," backward time =",self.timeBackward)
      sys.concatTime = sys.concatTime + (self.timeForward + self.timeBackward)
      self.timeForward =  0
      self.timeBackward = 0
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
