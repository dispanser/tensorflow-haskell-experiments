{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedLists #-}

module TFE.LinearRegression where

import           Data.ProtoLens                 ( decodeMessageOrDie )
import           Control.Monad                  ( forM_, replicateM_ )
import           GHC.Int                        ( Int64 )

import qualified Data.Vector                   as V
import qualified TensorFlow.Core               as TF
import qualified TensorFlow.GenOps.Core        as TFC
import qualified TensorFlow.Minimize           as TF
import qualified TensorFlow.Ops                as TF
                                         hiding ( initializedVariable )
import qualified TensorFlow.Variable           as TF
import qualified TensorFlow.Logging            as TFL

import qualified Readme                        as R

defaultLearnRate :: Float
defaultLearnRate = 0.00001

iterations :: Int64
iterations = 3000

-- reduce to enable logging
logEveryNth :: Int64
logEveryNth = 100000

simpleLinearRegression :: [Float] -> [Float] -> IO (Float, Float)
simpleLinearRegression = simpleLinearRegression' defaultLearnRate

-- | compute simple linear regression, using gradient descent on tensorflow
simpleLinearRegression' :: Float -> [Float] -> [Float] -> IO (Float, Float)
simpleLinearRegression' learningRate x y =
    TFL.withEventWriter "test.log" $ \eventWriter -> TF.runSession $ do
        let x' = TF.vector x
            y' = TF.vector y
        b0 <- TF.initializedVariable 0
        b1 <- TF.initializedVariable 0

        let yHat = (x' * TF.readValue b1) + TF.readValue b0
            loss = TFC.square $ yHat - y'

        TFL.histogramSummary "losses" loss
        TFL.scalarSummary "error" $ TF.reduceSum loss
        TFL.scalarSummary "intercept" $ TF.readValue b0
        TFL.scalarSummary "weight" $ TF.readValue b1

        trainStep <- TF.minimizeWith (TF.gradientDescent learningRate)
                                     loss
                                     [b0, b1]
        summaryT <- TFL.mergeAllSummaries
        forM_ ([1 .. iterations] :: [Int64]) $ \step -> do
            if step `mod` logEveryNth == 0
                then do
                   -- TF.run_ trainStep
                    ((), summaryBytes) <- TF.run (trainStep, summaryT)
                    (TF.Scalar beta0, TF.Scalar beta1) <- TF.run
                        (TF.readValue b0, TF.readValue b1)
                    -- liftIO $ putStrLn $ "Y  = " ++ show beta1 ++ "X + " ++ show beta0
                    let summary = decodeMessageOrDie (TF.unScalar summaryBytes)
                    TFL.logSummary eventWriter step summary
                else TF.run_ trainStep

        (TF.Scalar b0', TF.Scalar b1') <- TF.run (TF.readValue b0, TF.readValue b1)
        return (b0', b1')

-- | another linear regression solution, based on https://mmhaskell.com/blog/2017/8/14/starting-out-with-haskell-tensor-flow
simpleLinearRegressionMMH :: Float -> V.Vector Float -> V.Vector Float -> IO (Float, Float)
simpleLinearRegressionMMH learningRate xInput yInput = TF.runSession $ do
    let xSize = fromIntegral $ length xInput
    let ySize = fromIntegral $ length yInput
    (w :: TF.Variable Float       ) <- TF.initializedVariable 0
    (b :: TF.Variable Float       ) <- TF.initializedVariable 0
    (x :: TF.Tensor TF.Value Float) <- TF.placeholder [xSize]
    let linear_model = ((TF.readValue w) `TF.mul` x) `TF.add` (TF.readValue b)
    (y :: TF.Tensor TF.Value Float) <- TF.placeholder [ySize]
    let square_deltas = TFC.square (linear_model `TF.sub` y)
    let loss          = TF.reduceSum square_deltas
    trainStep <- TF.minimizeWith (TF.gradientDescent learningRate) loss [w, b]
    let trainWithFeeds =
            \xF yF -> TF.runWithFeeds [TF.feed x xF, TF.feed y yF] trainStep
    replicateM_ 1000
        (trainWithFeeds
            (TF.encodeTensorData (TF.Shape [xSize]) xInput)
            (TF.encodeTensorData (TF.Shape [ySize]) yInput))
    (TF.Scalar w_learned, TF.Scalar b_learned) <- TF.run (TF.readValue w, TF.readValue b)
    return (b_learned, w_learned)

-- | directly compute beta0, beta1. See ISLR, Chapter 3.1, p 62.
simpleLinearClosed :: [Float] -> [Float] -> IO (Float, Float)
simpleLinearClosed xs ys =
    let xMean = mean xs
        yMean = mean ys
        mean r = sum r / fromIntegral (length r)
        square r = r * r
        xYs   = sum $ zipWith (*) ((xMean -) <$> xs) ((yMean -) <$> ys)
        beta1 = xYs / sum ((\x -> square (x - xMean)) <$> xs)
        beta0 = yMean - xMean * beta1
    in  return (beta0, beta1)
