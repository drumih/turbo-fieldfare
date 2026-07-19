public struct AppPromptPreset: Identifiable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let prefix: String
    public let isPrimary: Bool

    public init(id: String, title: String, prefix: String, isPrimary: Bool) {
        self.id = id
        self.title = title
        self.prefix = prefix
        self.isPrimary = isPrimary
    }

    public static let all: [AppPromptPreset] = [
        AppPromptPreset(
            id: "paris",
            title: "Paris",
            prefix: "Paris is the capital and largest city of France. Situated on the River Seine, the city is known for",
            isPrimary: true),
        AppPromptPreset(
            id: "fibonacci",
            title: "Python Fibonacci",
            prefix: """
            An iterative Python function can return the first n Fibonacci numbers without recursion. It returns an empty list when n is zero, uses O(n) time, and includes a small correctness check:

            ```python
            def fibonacci(n: int) -> list[int]:
            """,
            isPrimary: true),
        AppPromptPreset(
            id: "fieldfare",
            title: "Meet the fieldfare",
            prefix: "The fieldfare (Turdus pilaris) is a grey-headed, chestnut-backed thrush that reaches Britain and Ireland from northern Europe each autumn. Its noisy winter flocks roam the countryside in search of berries, and",
            isPrimary: true),
        AppPromptPreset(
            id: "cosine-similarity",
            title: "Dot product and cosine similarity",
            prefix: """
            The unit vectors u = [1, 0] and v = [0, 1] are orthogonal. Their dot product is 0, both lengths are 1, and their cosine similarity is exactly 0. A compact check is:

            ```python
            assert dot([1, 0], [0, 1]) == 0
            assert cosine_similarity([1, 0], [0, 1]) ==
            """,
            isPrimary: false),
        AppPromptPreset(
            id: "recommendation-systems",
            title: "How recommendations learn",
            prefix: """
            # How Collaborative Filtering Learns Film Recommendations

            A film-streaming service records a sparse user–item matrix: rows represent viewers, columns represent films, and observed entries represent ratings or interactions. Most entries are missing because each viewer has seen only a small fraction of the catalog. Matrix factorization learns a low-dimensional vector for every viewer and every film. Their dot product estimates a missing preference score, so films with the largest estimated scores can be ranked for that viewer.

            The explanation has exactly four sections: the sparse interaction matrix; latent viewer and film vectors; training and prediction; and limitations and evaluation. A missing interaction is unknown, not a negative rating. Training adjusts the vectors to reduce error on observed interactions with regularization. Evaluation uses held-out interactions and ranking measures. The limitations are cold start and popularity bias. The fourth section ends the article without a summary, repeated title, or repeated instructions.

            ## The sparse interaction matrix
            """,
            isPrimary: false),
        AppPromptPreset(
            id: "matrix-multiplication",
            title: "Matrix multiplication",
            prefix: """
            # Matrix Multiplication: Why Rows Meet Columns

            For A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]], the complete verified calculation is:

            C[1,1] = 1 x 5 + 2 x 7 = 19
            C[1,2] = 1 x 6 + 2 x 8 = 22
            C[2,1] = 3 x 5 + 4 x 7 = 43
            C[2,2] = 3 x 6 + 4 x 8 = 50

            Thus C = AB = [[19, 22], [43, 50]]. The explanation below does not repeat or recompute those numeric equations. It has exactly three sections: why the inner dimensions match; how each output cell is a row-column dot product; and the general m x n by n x p rule. The final section states C[i,j] = sum over k of A[i,k]B[k,j] and then ends, without another example, code, references, or a summary.

            ## Why the inner dimensions match
            """,
            isPrimary: false),
    ]

    public static var primary: [AppPromptPreset] { all.filter(\.isPrimary) }
    public static var secondary: [AppPromptPreset] { all.filter { !$0.isPrimary } }
}
