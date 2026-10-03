import unittest
import numpy as np
from clip_shift_audit.core import audit, auroc, confidence, cosine_logits, fit_threshold, risk_coverage
from clip_shift_audit.cli import demo_data


class MetricsTest(unittest.TestCase):
    def test_auc_orientation_ties(self):
        self.assertEqual(auroc([.9,.8], [.1,.2]), 1.)
        self.assertEqual(auroc([.1,.2], [.9,.8]), 0.)
        self.assertEqual(auroc([.5,.5], [.5,.5]), .5)

    def test_auc_matches_pairwise_definition(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            a, b = rng.integers(4,size=7), rng.integers(4,size=9)
            expected = np.mean((a[:,None]>b).astype(float)+.5*(a[:,None]==b))
            self.assertAlmostEqual(auroc(a,b), expected)

    def test_small_calibration_accepts_all(self):
        self.assertIsNone(fit_threshold([.1,.2], .05))
        self.assertEqual(fit_threshold(np.arange(19), .05), 0.)

    def test_fixed_threshold_independent_of_test(self):
        text,cal,test,y = demo_data()
        a = audit(text,cal,test,y)
        b = audit(text,cal,test*0.5,y)
        for m in a["methods"]:
            self.assertEqual(a["methods"][m]["threshold"], b["methods"][m]["threshold"])

    def test_bad_embeddings(self):
        for data in [[[0,0]], [[float("nan"),1]], []]:
            with self.assertRaises(ValueError):
                cosine_logits(data, [[1,0]])

    def test_bad_label_types_and_range(self):
        x = list(demo_data())
        for y in [x[3].astype(float), np.full(len(x[3]),-2), x[3][:-1]]:
            with self.assertRaises(ValueError):
                audit(*x[:3],y)

    def test_stable_softmax(self):
        scores = confidence([[1,-1],[1,1]], "msp", 1e-6)
        np.testing.assert_allclose(scores,[1,.5])

    def test_ties_are_grouped(self):
        curve = risk_coverage(np.array([1.,1.,0.]), np.array([True,False,True]))
        self.assertEqual(len(curve),2)
        self.assertEqual(curve[0]["risk"],.5)

    def test_reproducibility_and_protocol(self):
        a,b = audit(*demo_data()),audit(*demo_data())
        self.assertEqual(a,b)
        self.assertEqual(a["n_unknown"],160)
        self.assertGreater(a["methods"]["cosine"]["auroc"],.9)
        self.assertTrue(all(0 <= m["fpr_at_95_tpr"] <= 1 for m in a["methods"].values()))


if __name__ == "__main__":
    unittest.main()
