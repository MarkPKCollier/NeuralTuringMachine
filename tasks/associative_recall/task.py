class AssociativeRecallTask:
    name = 'associative_recall'

    @staticmethod
    def offset(max_len_placeholder):
        return 3 * (max_len_placeholder + 1) + 2
