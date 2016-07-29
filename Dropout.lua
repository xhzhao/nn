local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(p,v1,inplace)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.inplace = inplace
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()

   self.timerEnable = sys.timerEnable
   self.timeForward = 0
   self.timeBackward = 0
   self.cnt = 0


end

function Dropout:updateOutput(input)
   startTime = sys.clock()
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   mid1 = sys.clock()
   if self.p > 0 then
      if self.train then
         self.noise:resizeAs(input)
         self.noise:bernoulli(1-self.p)
	 mid2 = sys.clock()
         if self.v2 then
            self.noise:div(1-self.p)
         end
         self.output:cmul(self.noise)
      elseif not self.v2 then
         self.output:mul(1-self.p)
      end
   end

   if self.timerEnable then
	mid3 = sys.clock()
	print("DropOut forward detail: mid1=",mid1-startTime,", mid2=",mid2-mid1,", mid3=",mid3-mid2)
                print("DropOut  forward time =         ",self.timeForward," backward time =",self.timeBackward)
                sys.dropTime = sys.dropTime + (self.timeForward + self.timeBackward)
        self.timeForward =  (sys.clock() - startTime)
        self.timeBackward = 0
        self.cnt = self.cnt + 1
   end



   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   startTime = sys.clock()
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

   if self.timerEnable then
        self.timeBackward =  (sys.clock() - startTime)
   end

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
