local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(p,v1,inplace,stochasticInference)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.inplace = inplace
   self.stochastic_inference = stochasticInference or false
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function Dropout:updateOutput(input)
   local start=sys.clock()
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.p > 0 then
      if self.train or self.stochastic_inference then
         self.noise:resizeAs(input)
         self.noise:bernoulli(1-self.p)
         if self.v2 then
            self.noise:div(1-self.p)
         end
         self.output:cmul(self.noise)
      elseif not self.v2 then
         self.output:mul(1-self.p)
      end
   end
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 500 then
      print("Dropout_F = ", self.t1)
      print("Dropout_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end

   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   local start=sys.clock()
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
      end
   else
      if not self.v2 and self.p > 0 then
         self.gradInput:mul(1-self.p)
      end
   end
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end

function Dropout:setp(p)
   self.p = p
end

function Dropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function Dropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
