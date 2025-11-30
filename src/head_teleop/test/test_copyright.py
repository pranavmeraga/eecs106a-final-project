from ament_copyright.main import main
import pytest

@pytest.mark.skip(reason='No copyright header required')
@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found errors'
