import caffe.proto.caffe_pb2 as caffe_pb2

import google.protobuf as protobuf

import collections
import sys


class ProtoObject(object):

    def __init__(self, msg_name, **kwargs):
        self._proto = getattr(caffe_pb2, msg_name)
        self._field_info = self._proto.DESCRIPTOR.fields_by_name
        self._passthrough = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_message(self):
        self._passthrough = False
        try:
            message = self._proto()
            for field, info in self._field_info.items():
                if not hasattr(self, field):
                    if info.label == info.LABEL_REQUIRED:
                        raise AttributeError('Net missing required field: {}'.format(field))
                    continue
                val = getattr(self, field)
                if info.label == info.LABEL_REPEATED:
                    if info.message_type:
                        sub_msgs = [sub.to_message() for sub in val]
                        getattr(message, field).extend(sub_msgs)
                    else:
                        getattr(message, field).extend(val)
                else:
                    if info.message_type:
                        getattr(message, field).CopyFrom(val.to_message())
                    else:
                        setattr(message, field, val)
        finally:
            self._passthrough = True
        return message

    @staticmethod
    def _message_type_to_protoclass(message_type):
        if message_type.name == 'NetParameter':
            return ProtoNet
        elif message_type.name == 'LayerParameter':
            return ProtoLayer
        else:
            return getattr(ProtoParam, message_type.name)

    @staticmethod
    def from_message(message):
        cls = ProtoObject._message_type_to_protoclass(message.DESCRIPTOR)
        protoobj = cls()
        protoobj._message = message
        msg_fields = [field[0].name for field in message.ListFields()]
        for field, info in protoobj._field_info.items():
            if field not in msg_fields:
                continue
            val = getattr(message, field)
            if info.label == info.LABEL_REPEATED:
                val = list(val)
            if info.message_type:
                if info.label == info.LABEL_REPEATED:
                    protocls = cls._message_type_to_protoclass(info.message_type)
                    setattr(protoobj, field, [ProtoObject.from_message(param_msg) for param_msg in val])
                else:
                    setattr(protoobj, field, ProtoObject.from_message(val))
            else:
                setattr(protoobj, field, val)
        return protoobj

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        elif name in self._field_info:
            info = self._field_info[name]
            # allow single value assignment to repeated fields
            if info.label == info.LABEL_REPEATED and not isinstance(value, (list, tuple)):
                value = [value]
            # convert dictionaries to ProtoObjects if necessary
            message_desc = self._field_info[name].message_type
            if message_desc:
                cls = self._message_type_to_protoclass(message_desc)
                if info.label == info.LABEL_REPEATED:
                    new_value = []
                    for el in value:
                        if isinstance(el, dict):
                            new_value.append(cls(**el))
                        else:
                            new_value.append(el)
                    value = new_value
                elif isinstance(value, dict):
                    value = cls(message_desc.name, **value)
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('Invalid field {}'.format(name))

    def __getattr__(self, name):
        if self._passthrough and name in self._field_info:
            default = self._field_info[name].default_value
            if default is not None:
                return default
        raise AttributeError('Invalid field {}'.format(name))


class ProtoNet(ProtoObject):

    def __init__(self, **kwargs):
        ProtoObject.__init__(self, 'NetParameter', **kwargs)

    @staticmethod
    def from_prototxt(inpath):
        message = caffe_pb2.NetParameter()
        with open(inpath, 'r') as f:
            protobuf.text_format.Merge(f.read(), message)
        return ProtoNet.from_message(message)

    def rename_blob(self, old, new):
        for layer in self.layer:
            if hasattr(layer, 'top'):
                for i, name in enumerate(layer.top):
                    if name == old:
                        layer.top[i] = new
            if hasattr(layer, 'bottom'):
                for i, name in enumerate(layer.bottom):
                    if name == old:
                        layer.bottom[i] = new

    def __getitem__(self, name):
        layers = []
        for layer in self.layer:
            if layer.name == name:
                layers.append(layer)
        if not layers:
            return None
        elif len(layers) == 1:
            return layers[0]
        else:
            result = {}
            for layer in layers:
                result[layer.include[0].phase] = layer
            return result

    def __delitem__(self, name):
        indices = []
        for i, layer in enumerate(self.layer):
            if layer.name == name:
                indices.append(i)
        for index in indices[::-1]:
            del self.layer[index]

    def write_prototxt(self, outpath):
        with open(outpath, 'w') as f:
            f.write(protobuf.text_format.MessageToString(self.to_message()))


class MetaProtoLayer(type):

    def __getattr__(self, name):
        return FakeLayerClass(name)


class ProtoLayer(ProtoObject):
    __metaclass__ = MetaProtoLayer

    def __init__(self, **kwargs):
        ProtoObject.__init__(self, 'LayerParameter', **kwargs)

    def __repr__(self):
        return '<ProtoLayer {} {}>'.format(self.type, self.name)

    def set_phase(self, phase):
        self.include = [ProtoParam.NetStateRule(phase=phase)]
        return self


class FakeLayerClass(object):

    def __init__(self, name):
        self._proto = getattr(caffe_pb2, 'LayerParameter')
        self.name = name

    def __call__(self, *args, **kwargs):
        return ProtoLayer(type=self.name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._proto, name)


class MetaProtoParam(type):

    def __getattr__(self, name):
        return FakeParamClass(name)


class FakeParamClass(object):

    def __init__(self, name):
        self._proto = getattr(caffe_pb2, name)
        self.name = name

    def __call__(self, *args, **kwargs):
        return ProtoParam(self.name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._proto, name)


class ProtoParam(ProtoObject):
    __metaclass__ = MetaProtoParam

    def __init__(self, msg_name, **kwargs):
        ProtoObject.__init__(self, msg_name, **kwargs)

    def __repr__(self):
        return '<ProtoParam {}>'.format(self._proto.DESCRIPTOR.name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        arg = sys.argv[1]
    else:
        arg = '/home/tzeng/src/caffe/models/bvlc_reference_caffenet/train_val.prototxt'
    net = ProtoNet.from_prototxt(arg)
    msg = net.to_message()
