import pytest

from finharness.config.crypto import SecretCipher, SecretCipherError
from finharness.config.store import (
    ActiveConfigDeleteError,
    ConfigNotFound,
    ConfigStore,
    DuplicateConfigName,
)

# Neutral placeholder values: these are fabricated strings for offline tests,
# never real credentials.
FAKE_A = "placeholder-value-a"
FAKE_B = "placeholder-value-b"
FAKE_C = "placeholder-value-c"
FAKE_D = "placeholder-value-d"


@pytest.fixture
def cipher(tmp_path):
    return SecretCipher(tmp_path / "secret.key")


@pytest.fixture
def store(tmp_path, cipher):
    return ConfigStore(tmp_path / "config.db", cipher=cipher)


def test_cipher_round_trips_and_persists_key(tmp_path):
    key_path = tmp_path / "secret.key"
    first = SecretCipher(key_path)

    token = first.encrypt(FAKE_A)

    assert token != FAKE_A.encode("utf-8")
    assert SecretCipher(key_path).decrypt(token) == FAKE_A


def test_cipher_rejects_tampered_token(cipher):
    token = bytearray(cipher.encrypt(FAKE_A))
    token[-1] ^= 0xFF

    with pytest.raises(SecretCipherError):
        cipher.decrypt(bytes(token))


def test_store_encrypts_key_on_disk(store, tmp_path):
    store.create(
        name="deepseek",
        kind="openai_compat",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        env_key=None,
        api_key=FAKE_A,
        activate=True,
    )

    raw = (tmp_path / "config.db").read_bytes()

    assert FAKE_A.encode("utf-8") not in raw


def test_store_creates_and_lists_redacted_records(store):
    created = store.create(
        name="deepseek",
        kind="openai_compat",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        env_key="DEEPSEEK_API_KEY",
        api_key=FAKE_A,
        activate=True,
    )

    assert created.id > 0
    assert created.has_key is True
    assert created.is_active is True
    assert "api_key" not in created.__dataclass_fields__
    assert store.count() == 1


def test_store_keeps_exactly_one_active_config(store):
    first = store.create(
        name="a", kind="openai_compat", base_url="https://a/v1", model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )
    second = store.create(
        name="b", kind="openai_compat", base_url="https://b/v1", model="m2",
        env_key=None, api_key=FAKE_B, activate=True,
    )

    assert store.get(first.id).is_active is False
    active = store.get_active()
    assert active is not None
    assert active.id == second.id


def test_store_activate_switches_the_single_active_row(store):
    first = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )
    second = store.create(
        name="b", kind="openai_compat", base_url=None, model="m2",
        env_key=None, api_key=FAKE_B, activate=False,
    )

    store.activate(second.id)

    assert store.get_active().id == second.id
    assert store.get(first.id).is_active is False


def test_store_update_without_key_preserves_the_secret(store):
    record = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_C, activate=True,
    )

    store.update(
        record.id, name="a2", kind="openai_compat", base_url="https://new/v1",
        model="m2", env_key=None, api_key=None,
    )

    assert store.resolve_key(record.id) == FAKE_C
    assert store.get(record.id).model == "m2"


def test_store_update_with_key_replaces_the_secret(store):
    record = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_C, activate=True,
    )

    store.update(
        record.id, name="a", kind="openai_compat", base_url=None,
        model="m1", env_key=None, api_key=FAKE_D,
    )

    assert store.resolve_key(record.id) == FAKE_D


def test_store_resolve_key_falls_back_to_environment(store, monkeypatch):
    monkeypatch.setenv("FH_TEST_KEY", FAKE_B)
    record = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key="FH_TEST_KEY", api_key=None, activate=True,
    )

    assert store.resolve_key(record.id) == FAKE_B


def test_store_rejects_duplicate_names(store):
    store.create(
        name="dup", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )

    with pytest.raises(DuplicateConfigName):
        store.create(
            name="dup", kind="openai_compat", base_url=None, model="m2",
            env_key=None, api_key=FAKE_A, activate=False,
        )


def test_store_refuses_to_delete_the_active_config_while_others_remain(store):
    first = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )
    store.create(
        name="b", kind="openai_compat", base_url=None, model="m2",
        env_key=None, api_key=FAKE_B, activate=False,
    )

    with pytest.raises(ActiveConfigDeleteError):
        store.delete(first.id)


def test_store_allows_deleting_a_non_active_config(store):
    first = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )
    second = store.create(
        name="b", kind="openai_compat", base_url=None, model="m2",
        env_key=None, api_key=FAKE_B, activate=False,
    )

    store.delete(second.id)

    assert [record.id for record in store.list_configs()] == [first.id]


def test_store_allows_deleting_the_last_config_even_if_active(store):
    only = store.create(
        name="a", kind="openai_compat", base_url=None, model="m1",
        env_key=None, api_key=FAKE_A, activate=True,
    )

    store.delete(only.id)

    assert store.count() == 0
    assert store.get_active() is None


def test_store_operations_on_missing_id_raise(store):
    with pytest.raises(ConfigNotFound):
        store.delete(999)
    with pytest.raises(ConfigNotFound):
        store.activate(999)
    with pytest.raises(ConfigNotFound):
        store.resolve_key(999)
