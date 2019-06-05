module TFE.LinearRegression where

import Control.Monad (replicateM_)
import qualified TensorFlow.Core               as TF
import qualified TensorFlow.GenOps.Core        as TF
import qualified TensorFlow.Minimize           as TF
import qualified TensorFlow.Ops                as TF
                                         hiding ( initializedVariable )
import qualified TensorFlow.Variable           as TF

-- | compute simple linear regression, using gradient descent on tensorflow
simpleLinearRegression :: [Float] -> [Float] -> IO (Float, Float)
simpleLinearRegression x y = TF.runSession $ do
    let x' = TF.vector x
        y' = TF.vector y
    b0 <- TF.initializedVariable 0
    b1 <- TF.initializedVariable 0

    let yHat = (x' `TF.mul` TF.readValue b1) `TF.add` TF.readValue b0
        loss = TF.square $ yHat `TF.sub` y'

    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.01) loss [b0, b1]
    replicateM_ 1000 (TF.run trainStep)

    (TF.Scalar b0', TF.Scalar b1') <- TF.run (TF.readValue b0, TF.readValue b1)
    return (b0', b1')
