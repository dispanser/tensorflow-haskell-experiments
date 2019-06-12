module Readme where

-- | Simple linear regression example for the README.

import           Control.Monad                  ( replicateM
                                                , replicateM_
                                                )
import           System.Random                  ( randomIO )
-- import           Test.HUnit                     ( assertBool )

import GHC.Exts (IsList, Item)
import qualified TensorFlow.Core               as TF
import qualified TensorFlow.GenOps.Core        as TF
import qualified TensorFlow.Minimize           as TF
import qualified TensorFlow.Ops                as TF
                                         hiding ( initializedVariable )
import qualified TensorFlow.Variable           as TF

-- fit :: Float -> V.Vector Float -> V.Vector Float -> IO (Float, Float)
fit :: Float -> [Float] -> [Float] -> IO (Float, Float)
fit learningRate xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent learningRate) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (b', w')

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- replicateM 100 randomIO
    let yData = [ x * 3 + 8 | x <- xData ]
    -- Fit linear regression model.
    (w, b) <- fit 0.001 xData yData
    putStrLn $ show b ++ " | " ++ show w
    -- assertBool "w == 3" (abs (3 - w) < 0.001)
    -- assertBool "b == 8" (abs (8 - b) < 0.001)

fanta :: Int -> Bool
fanta n = if even n then True else False

-- fanta2 :: Int -> Int
-- fanta2 n = True
