{-# LANGUAGE OverloadedLists #-}

module TFE.LinearRegressionTest where

import GHC.Exts (fromList)
import TFE.LinearRegression (simpleLinearRegression', simpleLinearRegression, simpleLinearClosed, simpleLinearRegressionMMH)
import Test.Hspec
import Text.CSV (parseCSVFromFile, Record)
import qualified Readme as R


-- switching between different implementations
-- fitFunction = R.fit
fitFunction = simpleLinearRegression'
-- fitFunction = simpleLinearRegressionMMH

linearRegressionSpec :: Spec
linearRegressionSpec = do
    -- n = 6 vs n = 7: PASS vs FAIL (beta0, beta1: NaN)
    linearRegressionTest     0.01 3 2 $ equidist 6 1 6
    linearRegressionTest     0.01 3 2 $ equidist 7 1 7

    -- however, both of these fail:
    linearRegressionTest     0.01 3 2 $ equidist 7 1 6
    linearRegressionTest     0.01 3 2 $ equidist 6 1 7

    -- n = 9 vs n = 10: PASS vs FAIL (beta0, beta1: NaN) (reduced learning rate)
    linearRegressionTest     0.005 3 2 $ equidist 9 1 6
    linearRegressionTest     0.005 3 2 $ equidist 10 1 7

    -- another one, different learning rate, but diverges with growing sample size
    linearRegressionTest     0.001 3 2 $ equidist 26 1 10
    linearRegressionTest     0.001 3 2 $ equidist 27 1 10

    -- n = 99 vs n = 100, ranging from -1 to 1: PASS vs FAIL (beta1 estimate = 0)
    linearRegressionTest     0.01 3 2 $ equidist 99  (-1) 1
    linearRegressionTest     0.01 3 2 $ equidist 100 (-1) 1


-- | produce a list of n values equally distributed over the range (minX, maxX)
equidist :: Int -> Float -> Float -> [Float]
equidist n minX maxX =
    let n'  = fromIntegral $ n - 1
        f k = ((n' - k) * minX + k*maxX) / n'
    in f <$> [0 .. n']

roughlyEqual :: (Num a, Ord a, Fractional a) => a -> a -> Bool
roughlyEqual expected actual = 0.01 > abs (expected - actual)

linearRegressionTest :: Float -> Float -> Float -> [Float] -> Spec
linearRegressionTest learnRate beta0 beta1 xs = do
    let ys = (\x -> beta1*x + beta0) <$> xs
    it ("linear regression on one variable, n = "  ++
        show (length xs) ++ ", range (" ++ show (head xs) ++ ", " ++ show (last xs) ++ ")") $ do
            (beta0Hat, beta1Hat) <- fitFunction learnRate (fromList xs) (fromList ys)
            beta0Hat `shouldSatisfy` roughlyEqual beta0
            beta1Hat `shouldSatisfy` roughlyEqual beta1


islrOLSSpec :: Spec
islrOLSSpec =
    describe "regressing sales onto the other variables from ISLR Chapter 3.1" $ do
        csv <- runIO $ parseCSVFromFile "data/Advertising.csv"
        case csv of
            Right adData -> do
                let salesCol = extractColumn 4 adData :: [Float]
                    tvCol    = extractColumn 1 adData :: [Float]
                it "computes the same coefficients for sales ~ TV" $ do
                    (beta0', beta1') <- simpleLinearClosed (take 3 tvCol) (take 3 salesCol)
                    (intercept, beta1) <- simpleLinearRegression (take 3 tvCol) (take 3 salesCol)
                    putStrLn $ "Y  = " ++ show beta1 ++ "X + " ++ show intercept
                    putStrLn $ "Y' = " ++ show beta1' ++ "X + " ++ show beta0'
                    print $ length salesCol
                    print $ length tvCol
                    -- beta0' `shouldSatisfy` roughlyEqual 7.0325
                    -- beta1' `shouldSatisfy` roughlyEqual 0.0475
                    intercept `shouldSatisfy` roughlyEqual beta0'
                    beta1     `shouldSatisfy` roughlyEqual beta1'
            Left  err    -> fail $ "csv error: " ++ show err

extractColumn :: Read a => Int -> [Record] -> [a]
extractColumn c rs = read .  (!! c) <$> tail (filter (/= [""]) rs)
