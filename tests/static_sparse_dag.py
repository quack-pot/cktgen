import dagmin
import unittest

class StaticSparseDAGTests(unittest.TestCase):
    def test_xor_2(self) -> None:
        model: dagmin.models.StaticSparseDAG = dagmin.models.StaticSparseDAG(dagmin.models.StaticSparseDAGCreateInfo())

        xor_data: list[list[float]] = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]

        xor_targets: list[list[float]] = [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
        
        model.train(dagmin.models.StaticSparseDAGTrainInfo(input_data=xor_data, output_data=xor_targets))

        for datum, target in zip(xor_data, xor_targets):
            output: list[float] = model.evaluate(datum)

            for out_value, target_value in zip(output, target):
                self.assertEqual(round(out_value), target_value)

if __name__ == "__main__":
    unittest.main()