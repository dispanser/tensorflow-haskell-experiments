{-# LANGUAGE OverloadedStrings #-}

import Test.Tasty
import Test.Tasty.Hspec (testSpec)
import TFE.LinearRegressionTest (linearRegressionSpec)

main :: IO ()
main = do
  ols <- testSpec "hspec tests" linearRegressionSpec
  let tests = testGroup "all tests" [ ols ]
  defaultMain tests

