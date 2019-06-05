{-# LANGUAGE OverloadedStrings #-}

import Test.Hspec
import Test.Tasty
import Test.Tasty.Hspec (testSpec)
import TFE.LinearRegressionTest (linearRegressionSpec)

main :: IO ()
main = do
  s <- testSpec "hspec tests" linearRegressionSpec
  let tests = testGroup "all tests" [ s ]
  defaultMain tests

