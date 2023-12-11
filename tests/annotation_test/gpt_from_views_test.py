from annotation.gpt_from_views import get_initial_annotation


def test_get_initial_annotation():
    anno, urls = get_initial_annotation("yysvdHrJ2q9ZS50mMWdkYezE6dF")
    assert anno["frontView"] == 3, f"Got {anno}"


if __name__ == "__main__":
    test_get_initial_annotation()
    print("DONE")
