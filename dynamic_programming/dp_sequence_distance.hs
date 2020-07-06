import Data.List
import Data.Maybe
import Control.Monad
import System.Environment

-- generic edit distance cost function
cost2d :: Char -> Char -> Int
cost2d c1 c2
	| c1 == c2  = 0	
	| otherwise = 1

sequenceDistance2d :: String -> String -> [[Int]]
sequenceDistance2d a b = mem
	where
		aux i 0 = i
		aux 0 j = j
		aux i j = minimum [d1 i j, d2 i j, d3 i j]
		d1 i j = (mem!!(i-1)!!(j-1)) + cost2d (a!!(i-1)) (b!!(j-1))
		d2 i j = mem!!(i-1)!!j + cost2d (a!!(i-1)) '-'
		d3 i j = mem!!i!!(j-1) + cost2d (b!!(j-1)) '-'
		mem = [[aux x y | y <- [0..length b]] | x <- [0..length a]]

main :: IO ()
main = do
	args <- getArgs
	if not $ length args == 2 then
		do
			putStrLn "Usage: ./dp_sequence_distance sequence1 sequence2"
			return ()
	else do
		let s1 = args !! 0
		let s2 = args !! 1
		let mem = editDistance s1 s2
		let d = mem !! (length s1) !! (length s2)
		putStrLn $ show d
