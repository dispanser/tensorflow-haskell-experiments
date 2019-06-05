module TFE.LinearRegressionTest where

import TFE.LinearRegression (simpleLinearRegression)
import Test.Hspec

linearRegressionSpec :: Spec
linearRegressionSpec =
    describe "simple linear regression" $
        it "compute correct values for sample size 2" $ do
            let x = [0.0, 1.0]
            let y = [3.0, 5.0]
            res <- simpleLinearRegression x y
            fst res `shouldSatisfy` roughly 3
            snd res `shouldSatisfy` roughly 2

roughly :: (Num a, Ord a, Fractional a) => a -> a -> Bool
roughly expected actual = 0.001 > abs (expected - actual)


