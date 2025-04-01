from pytest import fixture


def pytest_addoption(parser):
    parser.addoption("--target_path", action="store")


@fixture(scope="session")
def target_path(request):
    return request.config.getoption("--target_path")
