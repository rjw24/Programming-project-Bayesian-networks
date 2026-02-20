from probability4e import *

T, F = True, False


class Diagnostics:
    """Use a Bayesian network to diagnose between three lung diseases.

    Diseases (boolean variables in the Bayes net):
        - TB         (tuberculosis)
        - Cancer     (lung cancer)
        - Bronchitis

    Evidence variables (all boolean as required by the AIMA BayesNet class):
        - Asia      : True means the patient visited Asia
        - Smoking   : True means the patient is a smoker
        - Xray      : True means the X-ray result is Abnormal
        - Dyspnea   : True means dyspnea is Present
    """

    def __init__(self):
        self.bn = BayesNet([
            # Priors
            ('Asia', '', 0.01),
            ('Smoking', '', 0.5),

            # Diseases / intermediate nodes
            ('TB', 'Asia', {T: 0.05, F: 0.01}),

            # The standard "Asia" lung-disease network
            ('Cancer', 'Smoking', {T: 0.1, F: 0.01}),

            ('Bronchitis', 'Smoking', {T: 0.6, F: 0.3}),

            # Deterministic OR node
            ('TBorC', 'TB Cancer', {
                (T, T): 1.0,
                (T, F): 1.0,
                (F, T): 1.0,
                (F, F): 0.0,
            }),

            # Symptoms
            ('Xray', 'TBorC', {T: 0.99, F: 0.05}),
            ('Dyspnea', 'TBorC Bronchitis', {
                (T, T): 0.9,
                (T, F): 0.7,
                (F, T): 0.8,
                (F, F): 0.1,
            }),
        ])

    # ---------------------------------------------------------------------
    # Helpers for converting GUI strings to booleans (or None for NA)

    @staticmethod
    def _norm(x):
        return 'na' if x is None else str(x).strip().lower()

    @classmethod
    def _yes_no_na(cls, x):
        """Convert Yes/No/NA to True/False/None."""
        v = cls._norm(x)
        if v in ('na', ''):
            return None
        if v == 'yes':
            return True
        if v == 'no':
            return False
        raise ValueError(f"Invalid value '{x}'. Expected Yes, No, or NA.")

    @classmethod
    def _xray_na(cls, x):
        """Convert Abnormal/Normal/NA to True/False/None."""
        v = cls._norm(x)
        if v in ('na', ''):
            return None
        if v == 'abnormal':
            return True
        if v == 'normal':
            return False
        raise ValueError(f"Invalid value '{x}'. Expected Abnormal, Normal, or NA.")

    @classmethod
    def _dyspnea_na(cls, x):
        """Convert Present/Absent/NA to True/False/None."""
        v = cls._norm(x)
        if v in ('na', ''):
            return None
        if v == 'present':
            return True
        if v == 'absent':
            return False
        raise ValueError(f"Invalid value '{x}'. Expected Present, Absent, or NA.")

    # ---------------------------------------------------------------------

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        """Return [most_likely_disease_name, probability].

        Inputs are the GUI strings:
            visit_to_asia : Yes / No / NA
            smoking       : Yes / No / NA
            xray_result   : Abnormal / Normal / NA
            dyspnea       : Present / Absent / NA

        Output:
            ["TB"|"Cancer"|"Bronchitis", p] where 0 <= p <= 1
        """

        evidence = {}

        a = self._yes_no_na(visit_to_asia)
        if a is not None:
            evidence['Asia'] = a

        s = self._yes_no_na(smoking)
        if s is not None:
            evidence['Smoking'] = s

        x = self._xray_na(xray_result)
        if x is not None:
            evidence['Xray'] = x

        d = self._dyspnea_na(dyspnea)
        if d is not None:
            evidence['Dyspnea'] = d

        # Enumeration inference for each disease.
        p_tb = float(enumeration_ask('TB', evidence, self.bn)[True])
        p_cancer = float(enumeration_ask('Cancer', evidence, self.bn)[True])
        p_bronchitis = float(enumeration_ask('Bronchitis', evidence, self.bn)[True])

        probs = {
            'TB': p_tb,
            'Cancer': p_cancer,
            'Bronchitis': p_bronchitis,
        }

        best = max(probs, key=probs.get)
        return [best, probs[best]]