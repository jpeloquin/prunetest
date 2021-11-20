# def test_power_precedence():
#     expr1 = Expression(
#         [UnOp("-"), Symbol("λ_freeswell"), BinOp("^"), NumericValue("2", Number(2))]
#     )
#     expr2 = Expression(
#         [
#             UnOp("-"),
#             Expression(
#                 [Symbol("λ_freeswell"), BinOp("^"), NumericValue("2", Number(2))]
#             ),
#         ]
#     )
#     assert expr1.eval() == expr2.eval()
