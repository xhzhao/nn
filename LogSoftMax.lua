local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self._type = self.output:type()
   self:setEngine(1)
end

function LogSoftMax:updateOutput(input)
   local startTime = sys.clock()
   input.THNN.LogSoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   if self.timerEnable then
   		print("LogSoftMax forward time =         ,",self.timeForward," backward time =",self.timeBackward)
                sys.logsoftmaxTime_forward = sys.logsoftmaxTime_forward + self.timeForward
                sys.logsoftmaxTime_backward = sys.logsoftmaxTime_backward + self.timeBackward
                self.timeForward = (sys.clock() - startTime)
                self.cnt = self.cnt + 1
   end
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   local startTime = sys.clock()
   input.THNN.LogSoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   if self.timerEnable then
        self.timeBackward = (sys.clock() - startTime)
   end
   return self.gradInput
end
